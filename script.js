// ================================================================
// 1. CẤU HÌNH & BIẾN TOÀN CỤC (ĐỒNG BỘ THEO PAYLOAD MỚI)
// ================================================================
let eventLogs = [];
const currentUser = JSON.parse(localStorage.getItem('currentUser'));

if (!currentUser) {
    window.location.href = 'login.html';
}

// Cấu hình Broker MQTT qua WebSocket Cloud
const mqtt_url = 'wss://aecd780b1f264cadacf3a1ffb4c985d2.s1.eu.hivemq.cloud:8884/mqtt'; 
const options = {
    connectTimeout: 4000,
    clientId: 'Web_Client_' + Math.random().toString(16).substr(2, 8),
    username: 'SMART_3003',
    password: 'DOANTOTNGHIEP2025a',
};

const MQTT_TOPICS = ['esp32/data', 'smartdoor/recognition/#', 'smartdoor/system/status'];
const HISTORY_MAX_LOGS = 40;

let dataTimeout; 
const TIMEOUT_MS = 10000; // Thay đổi theo yêu cầu mới: Sau 10s không có dữ liệu -> Reset trang chính
let isSyncing = false;
let envChart = null;

// Định nghĩa mảng 8 ID thiết bị khớp giao diện
const devices = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "dev7", "dev8"];
const deviceLabels = ["Đèn Phòng Khách", "Đèn Phòng Ngủ", "Đèn Phòng Bếp", "Đèn Nhà Vệ Sinh", "Đèn Ngoài Trời", "Rèm Cửa", "Cổng Chính", "Quạt"];

// ================================================================
// 2. KHỞI TẠO HỆ THỐNG KHI TẢI TRANG
// ================================================================
function init() {
    updateUI('display-name', `Chào, ${currentUser.username} (${currentUser.role})`);

    // Tự động kết xuất danh sách switch thiết bị
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

    // Khởi chạy đồng hồ và lịch thực tế
    updateRealTimeCalendar();
    setInterval(updateRealTimeCalendar, 1000);

    // Khởi tạo đồ thị môi trường Chart.js
    const chartCanvas = document.getElementById('envChart');
    if (chartCanvas) {
        envChart = new Chart(chartCanvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Nhiệt độ (°C)', data: [], borderColor: '#e74c3c', tension: 0.3, fill: false },
                    { label: 'Độ ẩm (%)', data: [], borderColor: '#3498db', tension: 0.3, fill: false }
                ]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                }
            }
        });
    }

    // Chuyển đổi qua lại giữa các menu tab điều hướng
    const tabs = document.querySelectorAll('.tab-btn');
    const breadcrumb = document.getElementById('breadcrumb');

    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.target;
            const label = btn.innerText;
            const targetPanel = document.getElementById(tabId);
            
            if (targetPanel) {
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                targetPanel.classList.add('active');
                
                tabs.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                if (breadcrumb) breadcrumb.innerText = `Trang chủ / ${label}`;
            }
        });
    });

    // Quản lý và xử lý phân quyền phân hệ Quản trị (Admin)
    const adminShortcut = document.getElementById('admin-shortcut');
    const adminTabBtn = document.querySelector('.tab-btn[data-target="admin-panel"]');
    const adminPanel = document.getElementById('admin-panel');

    if (currentUser.role === 'admin') {
        if (adminShortcut) adminShortcut.style.display = 'inline-block';
        if (adminTabBtn) adminTabBtn.style.display = 'inline-block';
        if (adminShortcut && adminTabBtn) {
            adminShortcut.addEventListener('click', () => adminTabBtn.click());
        }
        updateUserTable();
    } else {
        if (adminShortcut) adminShortcut.remove();
        if (adminTabBtn) adminTabBtn.remove();
        if (adminPanel) adminPanel.remove();
    }

    updateDeviceCount();
    renderHistory();
    renderAlerts();
    connectMQTT();
}

document.addEventListener('DOMContentLoaded', init);

// ================================================================
// 3. XỬ LÝ NHẬN VÀ HIỂN THỊ DỮ LIỆU CẢM BIẾN TOÀN DIỆN
// ================================================================
const client = (typeof mqtt !== 'undefined') ? mqtt.connect(mqtt_url, options) : null;

function connectMQTT() {
    if (!client) return;

    client.on('connect', () => {
        console.log('MQTT Connected');
        updateUI('mqtt-last-topic-summary', 'Đã kết nối');
        updateHomeStatus('Hệ thống hoạt động bình thường', 'info');
        
        MQTT_TOPICS.forEach((topic) => {
            client.subscribe(topic, { qos: 1 });
        });
        updateAlertBox("Hệ thống MQTT: Trực tuyến (Online)");
        updateUI('home-camera-status', 'Online');
    });

    client.on('reconnect', () => {
        updateHomeStatus('MQTT đang kết nối lại...', 'warning');
    });

    client.on('offline', () => {
        updateHomeStatus('MQTT ngoại tuyến', 'danger');
        updateUI('mqtt-last-topic-summary', 'Ngoại tuyến');
    });

    client.on('message', (topic, payload) => {
        const messageStr = payload.toString();

        updateUI('mqtt-last-topic-debug', topic);
        updateUI('mqtt-last-payload', messageStr);

        try {
            const data = JSON.parse(messageStr);

            if (topic === 'esp32/data') {
                // Xóa và thiết lập lại bộ đếm thời gian chờ 5 giây (Timeout)
                clearTimeout(dataTimeout);
                dataTimeout = setTimeout(resetDashboardData, TIMEOUT_MS);

                // [1] Hiển thị Nhiệt độ & Độ ẩm
                if (data.temp !== undefined && data.humi !== undefined) {
                    updateSensorData(data.temp, data.humi);
                }

                // [2] Khớp dữ liệu cảm biến khí độc GAS mới nhất (gas_adc & gas_status)
                if (data.gas_adc !== undefined) {
                    updateUI('gas-val', data.gas_adc + " ADC");
                    updateUI('home-gas', data.gas_adc + " ADC");
                }
                if (data.gas_status !== undefined) {
                    updateAirStatus(data.gas_status);
                }

                // [3] Khớp cảm biến Ánh sáng mới nhất (light_adc & light_status)
                if (data.light_adc !== undefined) {
                    updateUI('light-val', data.light_adc + " ADC");
                }
                if (data.light_status !== undefined) {
                    const periodElem = document.getElementById('day-night-status');
                    if (periodElem) {
                        const isDay = (data.light_status === "NGAY");
                        periodElem.innerText = `Trạng thái: ${isDay ? "BAN NGÀY" : "BAN ĐÊM"}`;
                        periodElem.style.color = isDay ? "#ffce67" : "#1e90ff";
                    }
                }

                // [4] Khớp cảm biến Mưa mới nhất (rain_adc & rain_status)
                if (data.rain_status !== undefined) {
                    let rainStr = "Không mưa ☀️";
                    let isRaining = false;
                    
                    if (data.rain_status === "MUA_NHE" || data.rain_status === "MUA_TO") {
                        rainStr = data.rain_status === "MUA_TO" ? "MƯA LỚN 🌧️" : "Mưa nhỏ 🌦️";
                        isRaining = true;
                    } else if (data.rain_status === "KHO_RAO") {
                        rainStr = "Khô ráo ☀️";
                    }

                    updateUI('rain-status', rainStr);
                    const rainElem = document.getElementById('rain-status');
                    if (rainElem) {
                        rainElem.style.color = isRaining ? "#3498db" : "#2ecc71";
                    }
                }

                // [5] Đồng bộ hóa trạng thái nút bấm vật lý phản hồi lên Web
                if (data.devices && Array.isArray(data.devices)) {
                    isSyncing = true;
                    data.devices.forEach((state, i) => {
                        const checkbox = document.getElementById(devices[i]);
                        if (checkbox) checkbox.checked = (state === 1);
                    });
                    isSyncing = false;
                    updateDeviceCount();
                }
            }
            // Hệ thống nhận diện xử lý luồng Camera AI thông minh
            else if (topic === 'smartdoor/recognition/known') {
                const name = data.person_name || 'Người quen';
                updateUI('home-camera-status', 'Online');
                updateAlertBox(`✅ Người quen xuất hiện: ${name}`);
                pushEventLog('Nhận diện khuôn mặt', `Thành viên [${name}] quét thực thể mở khóa thành công.`, 'safe');
            }
            else if (topic === 'smartdoor/recognition/unknown') {
                updateAlertBox(`🚨 CẢNH BÁO: Phát hiện người lạ trước nhà!`);
                pushEventLog('An ninh cửa chính', `Có đối tượng không rõ danh tính di chuyển vùng camera.`, 'danger');
            }
            else if (topic === 'smartdoor/system/status') {
                const status = (data.status || 'offline').toLowerCase();
                updateUI('home-camera-status', status === 'online' ? 'Online' : 'Offline');
            }

        } catch (e) {
            console.error("Lỗi biên dịch gói tin JSON từ phần cứng:", e);
        }
    });
}

// ================================================================
// 4. TRUYỀN TÍN HIỆU ĐIỀU KHIỂN THIẾT BỊ (Mã lệnh: DK01 -> DK16)
// ================================================================
function onControlChange(deviceId, state) {
    if (isSyncing) return; 
    const index = devices.indexOf(deviceId);
    if (index < 0) return;

    const cmdNum = state ? (index + 1) : (index + 9);
    const command = "DK" + cmdNum.toString().padStart(2, '0');
    
    if (client && client.connected) {
        client.publish('esp32/commands', command, { qos: 1 });
        console.log("Lệnh đẩy xuống thành công:", command);
        pushEventLog('Điều khiển thiết bị', `Yêu cầu hệ thống phát mã lệnh ${command} tới [${deviceLabels[index]}].`, 'info');
    } else {
        alert("Lỗi phần cứng: Mất tín hiệu kết nối MQTT Broker!");
        document.getElementById(deviceId).checked = !state; 
    }
    updateDeviceCount();
}

function updateDeviceCount() {
    const total = devices.length;
    const active = devices.reduce((count, deviceId) => {
        const checkbox = document.getElementById(deviceId);
        return count + (checkbox && checkbox.checked ? 1 : 0);
    }, 0);
    updateUI('device-count', `${active}/${total}`);
}

// ================================================================
// 5. CÁC HÀM TIỆN ÍCH KẾT XUẤT CẢM BIẾN
// ================================================================
function updateUI(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value;
}

function updateSensorData(temp, humi) {
    updateUI('t-val', temp + "°C");
    updateUI('h-val', humi + "%");
    updateUI('home-temp', temp + "°C");
    updateUI('home-humi', humi + "%");
    
    if (envChart) {
        const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
        if (envChart.data.labels.length > 15) {
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

function updateAirStatus(statusStr) {
    const airStatus = document.getElementById('air-status');
    if (!airStatus) return;

    if (statusStr === "NGUY_HIEM") {
        airStatus.innerText = "🚨 NGUY HIỂM";
        updateHomeStatus("🚨 PHÁT HIỆN RÒ RỈ KHÍ GAS / NỒNG ĐỘ KHÓI ĐẠT MỨC NGUY HIỂM!", 'danger');
        pushEventLog('Báo động khẩn cấp', `Cảm biến Gas báo động đỏ trạng thái nguy hiểm nguy kịch.`, 'danger');
    } else if (statusStr === "CANH_BAO") {
        airStatus.innerText = "Cảnh báo";
        updateHomeStatus("⚠️ Cảnh báo: Chất lượng môi trường không khí phát hiện khói nhẹ.", 'warning');
        pushEventLog('Cảnh báo chất lượng', `Hệ thống ghi nhận nồng độ Gas tăng nhẹ đột biến.`, 'warning');
    } else {
        airStatus.innerText = "Trong lành";
        updateHomeStatus("Hệ thống hoạt động bình thường", 'safe');
    }
}

// ================================================================
// 6. PHÂN TÁCH NHẬT KÝ RA HAI TAB RIÊNG BIỆT KHÔNG BỊ TRỐNG
// ================================================================
function pushEventLog(title, details, type = 'info') {
    const entry = {
        time: new Date().toLocaleTimeString('vi-VN', { hour12: false }),
        title,
        details,
        type
    };
    eventLogs.unshift(entry);
    if (eventLogs.length > HISTORY_MAX_LOGS) eventLogs.pop();
    
    renderHistory();
    renderAlerts();
}

function renderHistory() {
    const container = document.getElementById('history-log');
    if (!container) return;

    if (eventLogs.length === 0) {
        container.innerHTML = '<div class="history-empty">Chưa có ghi nhận lịch sử hoạt động.</div>';
        return;
    }

    container.innerHTML = eventLogs.map(entry => `
        <div class="history-entry ${entry.type}">
            <div class="history-time">${entry.time}</div>
            <div class="history-content">
                <strong>${entry.title}</strong>
                <p>${entry.details}</p>
            </div>
        </div>
    `).join('');
}

function renderAlerts() {
    const container = document.getElementById('alerts-list');
    if (!container) return;

    const alerts = eventLogs.filter(item => ['danger', 'warning', 'safe'].includes(item.type));
    if (alerts.length === 0) {
        container.innerHTML = '<div class="alerts-empty">Hiện tại không ghi nhận cảnh báo khẩn cấp nào.</div>';
        return;
    }

    container.innerHTML = alerts.map(entry => `
        <div class="alert-entry ${entry.type}" style="padding:10px; margin-bottom:6px; border-radius:6px; background:rgba(255,255,255,0.02);">
            <div class="alert-time" style="font-weight:bold; color:#e74c3c;">⏱️ ${entry.time}</div>
            <div style="margin-top:4px; font-size:0.9rem;"><strong>${entry.title}:</strong> ${entry.details}</div>
        </div>
    `).join('');
}

function updateHomeStatus(message, level = 'info') {
    updateUI('home-status-pill', message);
    const pill = document.getElementById('home-status-pill');
    if (!pill) return;
    pill.className = `alert-pill ${level}`;
}

function updateAlertBox(msg) {
    updateUI('alert-box', msg);
}

// YÊU CẦU MỚI: Reset thông số trang chính nhưng GIỮ NGUYÊN HOÀN TOÀN Tab Lịch sử
function resetDashboardData() {
    // 1. Chỉ xóa các thẻ hiển thị giá trị tức thời trên màn hình chính và trang con
    ['home-temp', 'home-humi', 'home-gas', 't-val', 'h-val', 'gas-val', 'light-val', 'rain-status'].forEach(id => updateUI(id, "--"));
    
    const periodElem = document.getElementById('day-night-status');
    if (periodElem) periodElem.innerText = "--";
    
    updateUI('home-camera-status', 'Offline');
    updateHomeStatus('Mất kết nối: Quá 5s không nhận được dữ liệu từ ESP32.', 'danger');
    updateAlertBox("Hệ thống: Mất tín hiệu phần cứng ESP32 (Mất kết nối)");
    
    // Lưu ý: Không xóa hoặc can thiệp mảng `eventLogs`, giữ nguyên lịch sử cũ!
}

// ================================================================
// 7. ĐỒNG BỘ LỊCH CHUẨN THỜI GIAN THỰC ĐÚNG NĂM 2026
// ================================================================
function updateRealTimeCalendar() {
    const now = new Date();
    
    const monthYearStr = `Tháng ${now.getMonth() + 1}, ${now.getFullYear()}`;
    const monthElem = document.querySelector('.calendar-month');
    if (monthElem) monthElem.innerText = monthYearStr;

    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const fullDateStr = now.toLocaleDateString('vi-VN', options);
    const dayNameElem = document.querySelector('.calendar-day-name');
    if (dayNameElem) dayNameElem.innerText = fullDateStr;

    const dayNum = now.getDate().toString().padStart(2, '0');
    const dayShortName = now.toLocaleDateString('vi-VN', { weekday: 'short' });
    
    const dayBox = document.querySelector('.calendar-today-box span');
    const dayShortElem = document.querySelector('.calendar-today-box small');
    
    if (dayBox) dayBox.innerText = dayNum;
    if (dayShortElem) dayShortElem.innerText = dayShortName;

    document.querySelectorAll('.calendar-grid-real .day').forEach(dayDiv => {
        dayDiv.classList.remove('today');
        if (!dayDiv.classList.contains('inactive') && parseInt(dayDiv.innerText) === now.getDate()) {
            dayDiv.classList.add('today');
        }
    });
}

// ================================================================
// 8. ĐIỀU HÀNH TÀI KHOẢN (LOCALSTORAGE)
// ================================================================
function updateUserTable() {
    const tbody = document.getElementById('user-table-body');
    if (!tbody) return;

    const accounts = JSON.parse(localStorage.getItem('accounts')) || [];
    tbody.innerHTML = accounts.map(acc => `
        <tr>
            <td>${acc.username}</td>
            <td><span class="badge ${acc.role}">${acc.role.toUpperCase()}</span></td>
            <td>
                ${acc.username === 'admin' ? '<small style="color:#7f8c8d;">Mặc định</small>' : 
                `<button class="btn btn-danger" style="padding:2px 8px; font-size:0.8rem;" onclick="deleteUser('${acc.username}')">Xóa</button>`}
            </td>
        </tr>
    `).join('');
}

function updateAdminProfile() {
    const newU = document.getElementById('admin-new-u').value.trim();
    const newP = document.getElementById('admin-new-p').value.trim();

    if (!newU || !newP) {
        alert("Vui lòng điền thông tin tài khoản và mật khẩu!");
        return;
    }

    let accounts = JSON.parse(localStorage.getItem('accounts'));
    let adminAcc = accounts.find(acc => acc.username === currentUser.username);
    if (adminAcc) {
        adminAcc.username = newU;
        adminAcc.password = newP;
        currentUser.username = newU;
        currentUser.password = newP;
        localStorage.setItem('accounts', JSON.stringify(accounts));
        localStorage.setItem('currentUser', JSON.stringify(currentUser));
        alert("Cập nhật thông tin Admin thành công!");
        location.reload();
    }
}

function addNewUser() {
    const u = document.getElementById('user-u').value.trim();
    const p = document.getElementById('user-p').value.trim();

    if (!u || !p) {
        alert("Vui lòng nhập đủ thông tin tài khoản!");
        return;
    }

    let accounts = JSON.parse(localStorage.getItem('accounts')) || [];
    if (accounts.some(acc => acc.username === u)) {
        alert("Tên tài khoản này đã tồn tại!");
        return;
    }

    accounts.push({ username: u, password: p, role: 'user' });
    localStorage.setItem('accounts', JSON.stringify(accounts));
    alert(`Đã thêm thành công người dùng: ${u}`);
    document.getElementById('user-u').value = '';
    document.getElementById('user-p').value = '';
    updateUserTable();
}

function deleteUser(username) {
    if (username === 'admin') return;
    if (confirm(`Bạn có chắc chắn muốn xóa tài khoản [${username}]?`)) {
        let accounts = JSON.parse(localStorage.getItem('accounts')) || [];
        accounts = accounts.filter(acc => acc.username !== username);
        localStorage.setItem('accounts', JSON.stringify(accounts));
        updateUserTable();
    }
}
