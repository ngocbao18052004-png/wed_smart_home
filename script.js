// ================================================================
// 1. CẤU HÌNH & BIẾN TOÀN CỤC (ĐỒNG BỘ THEO PAYLOAD MỚI)
// ================================================================
let eventLogs = [];
let currentUser = null;

try {
    currentUser = JSON.parse(localStorage.getItem('currentUser'));
} catch (e) {
    console.error("Lỗi đọc thông tin người dùng từ localStorage:", e);
}

if (!currentUser) {
    window.location.href = 'login.html';
}

// Cấu hình Broker MQTT qua WebSocket Cloud
const mqtt_url = 'wss://aecd780b1f264cadacf3a1ffb4c985d2.s1.eu.hivemq.cloud:8884/mqtt'; 
const mqttOptions = {
    connectTimeout: 4000,
    clientId: 'Web_Client_' + Math.random().toString(16).substring(2, 10),
    username: 'SMART_3003',
    password: 'DOANTOTNGHIEP2025a',
};

const MQTT_TOPICS = ['esp32/data', 'smartdoor/recognition/#', 'smartdoor/system/status'];
const HISTORY_MAX_LOGS = 40;

let dataTimeout = null; 
const TIMEOUT_MS = 10000; // Sau 10s không có dữ liệu -> Reset trang chính
let isSyncing = false;
let envChart = null;
let calendarInterval = null;

// Khai báo mảng trạng thái toàn cục của 10 thiết bị để đồng bộ giao diện linh hoạt
let currentDeviceStates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// Định nghĩa mảng 10 ID thiết bị khớp giao diện
const devices = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "dev7", "dev8", "dev9", "dev10"];
const deviceLabels = [
    "Đèn Phòng Khách", // Index 0
    "Đèn Phòng Ngủ",   // Index 1
    "Đèn Phòng Bếp",   // Index 2
    "Đèn Nhà Vệ Sinh", // Index 3
    "Đèn Ngoài Trời",  // Index 4
    "Quạt phòng khách",// Index 5
    "Quạt phòng bếp",  // Index 6
    "Quạt phòng ngủ",  // Index 7
    "Mái Che Tự Động", // Index 8 (Dùng Switch)
    "Cổng Chính Servo" // Index 9 (Dùng Button + Đèn màu Xanh/Đỏ)
];

// ================================================================
// 2. KHỞI TẠO HỆ THỐNG KHI TẢI TRANG
// ================================================================
function init() {
    if (!currentUser) return;

    updateUI('display-name', `Chào, ${currentUser.username} (${currentUser.role})`);

    // Vẽ giao diện ban đầu (Sử dụng hàm render thông minh để tách biệt Switch và Button)
    renderDeviceUI(currentDeviceStates);

    // Khởi chạy đồng hồ và lịch thực tế (Xóa interval cũ nếu có để tránh trùng lặp)
    if (calendarInterval) clearInterval(calendarInterval);
    updateRealTimeCalendar();
    calendarInterval = setInterval(updateRealTimeCalendar, 1000);

    // Khởi tạo đồ thị môi trường Chart.js
    const chartCanvas = document.getElementById('envChart');
    if (chartCanvas && typeof Chart !== 'undefined') {
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
            // Tránh gán đè sự kiện nhiều lần lãng phí tài nguyên
            adminShortcut.onclick = () => adminTabBtn.click();
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

// HÀM TẠO HTML ĐỘNG CHO DANH SÁCH THIẾT BỊ (Tách biệt Mái Che và Cổng Chính)
function renderDeviceUI(states) {
    const container = document.getElementById('controls-container');
    if (!container) return;

    container.innerHTML = states.map((state, i) => {
        const id = devices[i];
        const isOn = (state === 1);

        // NẾU LÀ CỔNG CHÍNH (INDEX 9 - DÙNG BUTTON + ĐÈN TÍN HIỆU)
        if (i === 9) {
            return `
                <div class="control-item gate-item ${isOn ? 'gate-open' : 'gate-close'}" id="item-${id}" style="display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <span>${deviceLabels[i]}</span>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="gate-led" id="led-${id}" style="width: 14px; height: 14px; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px rgba(0,0,0,0.3); background-color: ${isOn ? '#ff4d4d' : '#2ecc71'};"></span>
                        <button class="btn-gate" onclick="onGateButtonClick('${id}', ${i})" style="background-color: #34495e; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 12px;">
                            ${isOn ? 'ĐÓNG CỔNG' : 'MỞ CỔNG'}
                        </button>
                    </div>
                </div>
            `;
        } 
        // TẤT CẢ CÁC THIẾT BỊ CÒN LẠI (BAO GỒM MÁI CHE INDEX 8 - DÙNG SWITCH GẠT)
        else {
            return `
                <div class="control-item" style="display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <span>${deviceLabels[i]}</span>
                    <label class="switch">
                        <input type="checkbox" id="${id}" ${isOn ? 'checked' : ''} onchange="onControlChange('${id}', this.checked)">
                        <span class="slider"></span>
                    </label>
                </div>
            `;
        }
    }).join('');
}

// ================================================================
// 3. XỬ LÝ NHẬN VÀ HIỂN THỊ DỮ LIỆU CẢM BIẾN TOÀN DIỆN
// ================================================================
const client = (typeof mqtt !== 'undefined') ? mqtt.connect(mqtt_url, mqttOptions) : null;

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
                // Xóa và thiết lập lại bộ đếm thời gian chờ
                if (dataTimeout) clearTimeout(dataTimeout);
                dataTimeout = setTimeout(resetDashboardData, TIMEOUT_MS);

                // [1] Hiển thị Nhiệt độ & Độ ẩm
                if (data.temp !== undefined && data.humi !== undefined) {
                    updateSensorData(data.temp, data.humi);
                }

                // [2] Khớp dữ liệu cảm biến khí độc GAS mới nhất
                if (data.gas_adc !== undefined) {
                    updateUI('gas-val', data.gas_adc + " ADC");
                    updateUI('home-gas', data.gas_adc + " ADC");
                }
                if (data.gas_status !== undefined) {
                    updateAirStatus(data.gas_status);
                }

                // [3] Khớp cảm biến Ánh sáng mới nhất
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

                // [4] Khớp cảm biến Mưa mới nhất
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

                // =========================================================================
                // ĐOẠN ĐỒNG BỘ: BÓC TÁCH "gate_log" CHUYỂN TIẾP LÊN TAB "AN NINH CỔNG"
                // =========================================================================
                if (data.gate_log !== undefined && data.gate_log !== "") {
                    // 1. Đồng bộ văn bản lên Tab "An Ninh Cổng" vừa tạo
                    const tabLogBox = document.getElementById('slave-gate-log-box');
                    if (tabLogBox) {
                        tabLogBox.innerText = data.gate_log;
                        
                        // Đổi màu thông báo trực quan theo kết quả xử lý của Slave
                        if (data.gate_log.includes("SAI") || data.gate_log.includes("THẤT BẠI") || data.gate_log.includes("KHONG_HOP_LE")) {
                            tabLogBox.style.color = '#ff4d4d'; // Đỏ nếu lỗi
                        } else if (data.gate_log.includes("THÀNH CÔNG") || data.gate_log.includes("HỢP LỆ")) {
                            tabLogBox.style.color = '#2ecc71'; // Xanh nếu OK
                        } else {
                            tabLogBox.style.color = '#ecf0f1'; // Trắng mặc định
                        }
                    }

                    // 2. Tự động ghi nhận lịch sử tương tác này vào hệ thống của Web để lưu vết
                    if (data.gate_log.includes("KHONG_HOP_LE") || data.gate_log.includes("sai") || data.gate_log.includes("THẤT BẠI")) {
                        pushEventLog('An ninh Cổng chính', `Cảnh báo từ Slave: ${data.gate_log}`, 'danger');
                    } else {
                        pushEventLog('Nhật ký Cổng chính', `Báo cáo từ Slave: ${data.gate_log}`, 'safe');
                    }
                }

                // [5] ĐỒNG BỘ HÓA TRẠNG THÁI NÚT BẤM VẬT LÝ VÀ PHẢN HỒI LÊN WEB
                if (data.devices && Array.isArray(data.devices)) {
                    isSyncing = true;
                    
                    currentDeviceStates = [...data.devices];

                    currentDeviceStates.forEach((state, i) => {
                        const id = devices[i];
                        const isOn = (state === 1);

                        if (i === 9) {
                            // A. Cập nhật giao diện cũ (nếu còn)
                            const led = document.getElementById(`led-${id}`);
                            const btn = document.querySelector(`#item-${id} .btn-gate`);
                            if (led) led.style.backgroundColor = isOn ? '#ff4d4d' : '#2ecc71';
                            if (btn) btn.innerText = isOn ? 'ĐÓNG CỔNG' : 'MỞ CỔNG';

                            // B. ĐỒNG BỘ SANG CÁC THÀNH PHẦN MỚI TRÊN TAB "AN NINH CỔNG"
                            const cardLed = document.getElementById('card-gate-led');
                            const cardStatusText = document.getElementById('card-gate-status-text');
                            const cardBtn = document.getElementById('card-btn-gate-control');

                            if (cardLed) cardLed.style.backgroundColor = isOn ? '#ff4d4d' : '#2ecc71';
                            if (cardStatusText) {
                                cardStatusText.innerText = isOn ? 'ĐANG MỞ' : 'ĐÃ ĐÓNG AN TOÀN';
                                cardStatusText.style.color = isOn ? '#ff4d4d' : '#2ecc71';
                            }
                            if (cardBtn) {
                                cardBtn.disabled = false;
                                cardBtn.style.opacity = '1';
                                cardBtn.innerText = isOn ? 'ĐÓNG CỔNG' : 'MỞ CỔNG';
                                cardBtn.style.backgroundColor = isOn ? '#e74c3c' : '#2ecc71';
                            }
                        } else {
                            const checkbox = document.getElementById(id);
                            if (checkbox) checkbox.checked = isOn;
                        }
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
                if (client && client.connected)
                {
                    client.publish(
                        'esp32/commands',
                        'DK19',
                        { qos: 1 }
                    );

                    console.log(
                        '[AI] Người quen được nhận diện -> Gửi DK19 mở cổng'
                    );
                }
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
// 4. TRUYỀN TÍN HIỆU ĐIỀU KHIỂN THIẾT BỊ (ĐÃ CẬP NHẬT MÃ LỆNH MỚI)
// ================================================================

// 4.1. Dành cho 9 thiết bị dạng SWITCH gạt thông thường (Bao gồm Mái che)
function onControlChange(deviceId, state) {
    if (isSyncing) return; 
    const index = devices.indexOf(deviceId);
    if (index < 0 || index === 9) return; 

    let command = "";

    if (index === 8) {
        command = state ? "DK17" : "DK18";
    } else {
        const cmdNum = state ? (index + 1) : (index + 9);
        command = "DK" + cmdNum.toString().padStart(2, '0');
    }
    
    sendMqttCommand(command, index, deviceId, !state, 'switch');
}

// 4.2. Dành riêng cho CỔNG CHÍNH SERVO (Kích hoạt từ cả Trang chủ và Tab An Ninh Cổng)
function onGateButtonClick(deviceId, index) {
    if (isSyncing) return;
    if (index !== 9) return;

    const currentState = currentDeviceStates[index];
    let command = (currentState === 0) ? "DK19" : "DK20";

    sendMqttCommand(command, index, deviceId, currentState, 'button');
}

// 4.3. CHỨC NĂNG MỚI: GỬI LỆNH THAY ĐỔI MẬT KHẨU TỪ TAB XUỐNG SLAVE
function changeGatePasswordMQTT() {
    const oldPin = document.getElementById('input-old-pin').value.trim();
    const newPin = document.getElementById('input-new-pin').value.trim();

    if (oldPin.length < 4 || newPin.length < 4) {
        alert("Mật khẩu phải đạt độ dài từ 4 đến 6 ký tự!");
        return;
    }
    if (isNaN(oldPin) || isNaN(newPin)) {
        alert("Mật khẩu bắt buộc phải là ký tự số (0-9)!");
        return;
    }

    if (client && client.connected) {
        // Đóng gói cấu trúc định dạng chuỗi: PASS:mật_khẩu_cũ,mật_khẩu_mới
        const payloadStr = `PASS:${oldPin},${newPin}`;
        
        client.publish('esp32/commands', payloadStr, { qos: 1 });
        console.log("👉 Đã phát lệnh cấu hình mật khẩu mật:", payloadStr);
        
        pushEventLog('Yêu cầu cấu hình', 'Đang truyền chuỗi khóa mật khẩu mới xuống phần cứng...', 'warning');
        
        document.getElementById('input-old-pin').value = '';
        document.getElementById('input-new-pin').value = '';
        
        const logBox = document.getElementById('slave-gate-log-box');
        if (logBox) {
            logBox.innerText = '🔄 Đang gửi dữ liệu và đợi xác nhận phản hồi từ Slave...';
            logBox.style.color = '#f1c40f'; 
        }
    } else {
        alert("Mất kết nối MQTT Broker! Không thể cấu hình từ xa vào lúc này.");
    }
}

// Hàm bổ trợ gửi lệnh tập trung để tránh lặp code
function sendMqttCommand(command, index, deviceId, rollbackState, type) {
    if (client && client.connected) {
        client.publish('esp32/commands', command, { qos: 1 });
        console.log("Lệnh đẩy xuống thành công:", command);
        pushEventLog('Điều khiển thiết bị', `Yêu cầu hệ thống phát mã lệnh ${command} tới [${deviceLabels[index]}].`, 'info');
        
        // Cập nhật giả lập tức thời giao diện để tăng trải nghiệm nhạy bén
        if (type === 'button') {
            currentDeviceStates[index] = (currentDeviceStates[index] === 1) ? 0 : 1;
            const isOn = (currentDeviceStates[index] === 1);
            
            // Đổi nhanh UI ở Trang chủ
            const led = document.getElementById(`led-${deviceId}`);
            const btn = document.querySelector(`#item-${deviceId} .btn-gate`);
            if (led) led.style.backgroundColor = isOn ? '#ff4d4d' : '#2ecc71';
            if (btn) btn.innerText = isOn ? 'ĐÓNG CỔNG' : 'MỞ CỔNG';

            // Đổi nhanh UI ở Tab "An Ninh Cổng" mới (Hiển thị trạng thái chờ màu Vàng)
            const cardLed = document.getElementById('card-gate-led');
            const cardStatusText = document.getElementById('card-gate-status-text');
            const cardBtn = document.getElementById('card-btn-gate-control');

            if (cardStatusText) {
                cardStatusText.innerText = isOn ? 'ĐANG MỞ...' : 'ĐANG ĐÓNG...';
                cardStatusText.style.color = '#f1c40f';
            }
            if (cardLed) {
                cardLed.style.backgroundColor = '#f1c40f';
                cardLed.style.boxShadow = '0 0 10px rgba(241, 196, 15, 0.5)';
            }
            if (cardBtn) {
                cardBtn.disabled = true;
                cardBtn.style.opacity = '0.5';
            }
        }
    } else {
        alert("Lỗi phần cứng: Mất tín hiệu kết nối MQTT Broker!");
        if (type === 'switch') {
            const checkbox = document.getElementById(deviceId);
            if (checkbox) checkbox.checked = rollbackState; 
        }
    }
    updateDeviceCount();
}

function updateDeviceCount() {
    const total = devices.length;
    let active = 0;

    currentDeviceStates.forEach((state) => {
        if (state === 1) active++;
    });

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
    // Ép kiểu dữ liệu về dạng số thực (float) để tránh dữ liệu lỗi từ chuỗi
    const parsedTemp = parseFloat(temp);
    const parsedHumi = parseFloat(humi);

    updateUI('t-val', parsedTemp + "°C");
    updateUI('h-val', parsedHumi + "%");
    updateUI('home-temp', parsedTemp + "°C");
    updateUI('home-humi', parsedHumi + "%");
    
    if (envChart) {
        const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
        if (envChart.data.labels.length > 15) {
            envChart.data.labels.shift();
            envChart.data.datasets[0].data.shift();
            envChart.data.datasets[1].data.shift();
        }
        envChart.data.labels.push(now);
        envChart.data.datasets[0].data.push(parsedTemp);
        envChart.data.datasets[1].data.push(parsedHumi);
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

function resetDashboardData() {
    ['home-temp', 'home-humi', 'home-gas', 't-val', 'h-val', 'gas-val', 'light-val', 'rain-status'].forEach(id => updateUI(id, "--"));
    
    const periodElem = document.getElementById('day-night-status');
    if (periodElem) periodElem.innerText = "--";
    
    updateUI('home-camera-status', 'Offline');
    updateHomeStatus('Mất kết nối: Quá 10s không nhận được dữ liệu từ ESP32.', 'danger');
    updateAlertBox("Hệ thống: Mất tín hiệu phần cứng ESP32 (Mất kết nối)");
}

// ================================================================
// 7. ĐỒNG BỘ LỊCH CHUẨN THỜI GIAN THỰC ĐÚNG NĂM 2026
// ================================================================
function updateRealTimeCalendar() {
    const now = new Date();
    
    const monthYearStr = `Tháng ${now.getMonth() + 1}, ${now.getFullYear()}`;
    const monthElem = document.querySelector('.calendar-month');
    if (monthElem) monthElem.innerText = monthYearStr;

    // Định nghĩa cục bộ để tránh ghi đè biến cấu hình MQTT options bên trên
    const localeOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const fullDateStr = now.toLocaleDateString('vi-VN', localeOptions);
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

    let accounts = [];
    try {
        accounts = JSON.parse(localStorage.getItem('accounts')) || [];
    } catch (e) {
        console.error("Lỗi parse mảng dữ liệu accounts:", e);
    }

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

    let accounts = [];
    try {
        accounts = JSON.parse(localStorage.getItem('accounts')) || [];
    } catch (e) {
        console.error(e);
    }

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

    let accounts = [];
    try {
        accounts = JSON.parse(localStorage.getItem('accounts')) || [];
    } catch (e) {
        accounts = [];
    }

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
        let accounts = [];
        try {
            accounts = JSON.parse(localStorage.getItem('accounts')) || [];
        } catch (e) {
            accounts = [];
        }
        accounts = accounts.filter(acc => acc.username !== username);
        localStorage.setItem('accounts', JSON.stringify(accounts));
        updateUserTable();
    }
}
