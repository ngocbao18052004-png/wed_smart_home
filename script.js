// ================================================================
// 1. CẤU HÌNH & BIẾN TOÀN CỤC (ĐÃ CẬP NHẬT NGƯỠNG TEST)
// ================================================================
let eventLogs = [];
const currentUser = JSON.parse(localStorage.getItem('currentUser'));
if (!currentUser) window.location.href = 'login.html';

// --- CẤU HÌNH NGƯỠNG XỬ LÝ TRÊN WEB ---
const GAS_THRESHOLD_DANGER = 350;   // Hạ thấp để dễ test (chỉ cần thổi nhẹ khói)
const GAS_THRESHOLD_WARNING = 250;  // Ngưỡng cảnh báo nhẹ
const LDR_DAY_THRESHOLD = 3600;     // Trên 600 là Ngày, dưới là Đêm

const mqtt_url = 'wss://aecd780b1f264cadacf3a1ffb4c985d2.s1.eu.hivemq.cloud:8884/mqtt'; 
const options = {
    connectTimeout: 4000,
    clientId: 'Web_Client_' + Math.random().toString(16).substr(2, 8),
    username: 'SMART_3003',
    password: 'DOANTOTNGHIEP2025a',
};

let dataTimeout; 
const TIMEOUT_MS = 15000;
let isSyncing = false;
let envChart;

const devices = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "dev7", "dev8"];
const deviceLabels = ["Đèn Phòng Khách", "Đèn Phòng Ngủ", "Đèn Phòng Bếp", "Đèn Nhà Vệ Sinh", "Đèn Ngoài Trời", "Rèm Cửa", "Cổng Chính", "Quạt"];

// ================================================================
// 2. KẾT NỐI MQTT & XỬ LÝ DỮ LIỆU
// ================================================================
const client = (typeof mqtt !== 'undefined') ? mqtt.connect(mqtt_url, options) : null;

if (client) {
    client.on('connect', () => {
        console.log('MQTT Connected');
        client.subscribe('esp32/data');
        updateAlertBox("Hệ thống: Trực tuyến (Online)");
    });

    client.on('message', (topic, payload) => {
        if (topic !== 'esp32/data') return;
        clearTimeout(dataTimeout);

        try {
            const data = JSON.parse(payload.toString());
            
            // --- Xử lý Ánh sáng & Ngày Đêm (Xử lý trực tiếp trên Web) ---
            if (data.light !== undefined) {
                updateUI('light-val', data.light + " lux");
                const period = (data.light < LDR_DAY_THRESHOLD) ? "Ngày" : "Đêm";
                const periodElem = document.getElementById('day-night-status');
                if (periodElem) {
                    periodElem.innerText = `Trạng thái: ${period}`;
                    periodElem.style.color = (period === "Ngày") ? "#ffce67" : "#1e90ff";
                }
            }

            // --- Xử lý GAS/MQ135 (Dùng ngưỡng mới để dễ test) ---
            if (data.gas !== undefined) {
                updateUI('gas-val', data.gas + " ppm");
                updateAirStatus(data.gas);
            }

            // --- Xử lý Nhiệt độ/Độ ẩm ---
            if (data.temp !== undefined && data.humi !== undefined) {
                updateSensorData(data.temp, data.humi);
            }

            // --- Xử lý Mưa ---
            if (data.rain !== undefined) {
                updateUI('rain-status', (data.rain == 1) ? "Đang mưa" : "Không mưa");
            }

            // --- Đồng bộ trạng thái Switch ---
            if (data.devices && Array.isArray(data.devices)) {
                isSyncing = true;
                data.devices.forEach((st, i) => {
                    const checkbox = document.getElementById(devices[i]);
                    if (checkbox) checkbox.checked = (st === 1);
                });
                isSyncing = false;
            }

        } catch(e) {
            console.error("Lỗi dữ liệu JSON:", e);
        }

        dataTimeout = setTimeout(resetDashboardData, TIMEOUT_MS);
    });
}

// ================================================================
// 3. CÁC HÀM XỬ LÝ GIAO DIỆN & LOGIC
// ================================================================

function updateUI(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value;
}

function updateSensorData(temp, humi) {
    updateUI('t-val', temp + "°C");
    updateUI('h-val', humi + "%");
    
    if (envChart) {
        const now = new Date().toLocaleTimeString();
        if(envChart.data.labels.length > 15) {
            envChart.data.labels.shift();
            envChart.data.datasets[0].data.shift();
            envChart.data.datasets[1].data.shift();
        }
        envChart.data.labels.push(now);
        envChart.data.datasets[0].data.push(temp);
        envChart.data.datasets[1].data.push(humi);
        envChart.update();
    }
}

function updateAirStatus(gasValue) {
    const airStatus = document.getElementById('air-status');
    if (!airStatus) return;
    airStatus.classList.remove("good", "warning", "danger");

    if (gasValue > GAS_THRESHOLD_DANGER) {
        airStatus.innerText = "NGUY HIỂM";
        airStatus.classList.add("danger");
        updateAlertBox("🚨 CẢNH BÁO: Phát hiện khí Gas/Ô nhiễm nặng!");
    } else if (gasValue > GAS_THRESHOLD_WARNING) {
        airStatus.innerText = "Kém";
        airStatus.classList.add("warning");
        updateAlertBox("⚠️ Cảnh báo: Chất lượng không khí thấp");
    } else {
        airStatus.innerText = "Tốt";
        airStatus.classList.add("good");
    }
}

function onControlChange(deviceId, state) {
    if (isSyncing) return;
    const index = devices.indexOf(deviceId);
    if (index < 0) return;

    const cmdNum = state ? (index + 1) : (index + 9);
    const command = "DK" + cmdNum.toString().padStart(2, '0');
    
    if (client && client.connected) {
        client.publish('esp32/commands', command);
        console.log("Lệnh gửi:", command);
    }
}

function updateAlertBox(msg) {
    updateUI('alert-box', msg);
}

function resetDashboardData() {
    ['t-val', 'h-val', 'gas-val', 'light-val', 'rain-status'].forEach(id => updateUI(id, "--"));
    const periodElem = document.getElementById('day-night-status');
    if (periodElem) periodElem.innerText = "--";
    updateAlertBox("Hệ thống: Mất kết nối ESP32 (Timeout)");
}

// ================================================================
// 4. KHỞI TẠO
// ================================================================
function init() {
    // 1. Hiển thị tên người dùng
    updateUI('display-name', `Chào, ${currentUser.username} (${currentUser.role})`);

    // 2. Ẩn khung Camera
    const streamContainer = document.getElementById('stream-container') || document.querySelector('.camera-section');
    if (streamContainer) streamContainer.style.display = 'none';

    // 3. Vẽ danh sách thiết bị điều khiển
    const container = document.getElementById('controls-container');
    if (container) {
        container.innerHTML = devices.map((id, i) => `
            <div class="control-item">
                <span>${deviceLabels[i]}</span>
                <label class="switch">
                    <input type="checkbox" id="${id}" onchange="onControlChange('${id}', this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        `).join('');
    }

    // 4. KIỂM TRA PHÂN QUYỀN VÀ CHẶN ADMIN (MỚI)
    if (currentUser.role !== 'admin') {
        // Xóa nút Admin trên Header
        const adminBtn = document.getElementById('admin-shortcut');
        if (adminBtn) adminBtn.remove();

        // Xóa nút Tab Quản trị trong Menu
        const adminTabBtn = document.querySelector('.tab-btn[data-target="admin-panel"]');
        if (adminTabBtn) adminTabBtn.remove();

        // Xóa luôn nội dung Panel Quản trị để không thể truy cập bằng bất cứ cách nào
        const adminPanel = document.getElementById('admin-panel');
        if (adminPanel) adminPanel.remove();
        
        console.log("Hệ thống: Đã chặn quyền truy cập Quản trị.");
    }

    // 5. Khởi tạo Biểu đồ Chart.js
    const chartCanvas = document.getElementById('envChart');
    if (chartCanvas) {
        envChart = new Chart(chartCanvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Nhiệt độ', data: [], borderColor: 'red', tension: 0.3, fill: false },
                    { label: 'Độ ẩm', data: [], borderColor: 'blue', tension: 0.3, fill: false }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    // 6. Logic chuyển Tab
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.target;
            
            // Chỉ cho phép chuyển tab nếu Panel đó còn tồn tại trong DOM
            const targetPanel = document.getElementById(tabId);
            if (targetPanel) {
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                targetPanel.classList.add('active');
                
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', init);
function updateRealTimeCalendar() {
    const now = new Date();
    
    // 1. Cập nhật Tháng/Năm
    const monthYearStr = `Tháng ${now.getMonth() + 1}, ${now.getFullYear()}`;
    const monthElem = document.querySelector('.calendar-month');
    if (monthElem) monthElem.innerText = monthYearStr;

    // 2. Cập nhật Thứ, Ngày Tháng Năm đầy đủ
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const fullDateStr = now.toLocaleDateString('vi-VN', options);
    const dayNameElem = document.querySelector('.calendar-day-name');
    if (dayNameElem) dayNameElem.innerText = fullDateStr;

    // 3. Cập nhật ô số ngày lớn (Box bên phải)
    const dayNum = now.getDate().toString().padStart(2, '0');
    const dayShortName = now.toLocaleDateString('vi-VN', { weekday: 'short' });
    
    const dayBox = document.querySelector('.calendar-today-box span');
    const dayShortElem = document.querySelector('.calendar-today-box small');
    
    if (dayBox) dayBox.innerText = dayNum;
    if (dayShortElem) dayShortElem.innerText = dayShortName;

    // 4. Highlight ngày hiện tại trong lưới lịch (Calendar Grid)
    // Xóa class 'today' cũ và gán cho ô có text khớp với ngày hiện tại
    document.querySelectorAll('.calendar-grid-real .day').forEach(dayDiv => {
        dayDiv.classList.remove('today');
        if (!dayDiv.classList.contains('inactive') && dayDiv.innerText == now.getDate()) {
            dayDiv.classList.add('today');
        }
    });
}

// Gọi khởi tạo trong hàm init() và chạy lặp lại
setInterval(updateRealTimeCalendar, 1000);

