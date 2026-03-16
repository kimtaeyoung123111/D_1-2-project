# 로봇 비전 기반 다층 적재 시스템 (Pick2Pack 3D Pick-and-Place)
> **조 이름:** D-1 - ROKEY  
> **팀원:** 이우현, 윤승현, 김태영, 이정환

본 프로젝트는 RealSense 뎁스 카메라와 YOLO 비전 모델을 융합하여, 컨베이어 벨트 위의 상품을 인식하고 두산 협동로봇(m0609)을 통해 박스 내부에 다층(Multi-layer)으로 적재하는 자동화 시스템입니다.  
시스템은 크게 **[로봇/비전 제어 PC]**와 **[웹/음성인식(STT)/컨베이어 제어 허브 PC]**로 나뉘어 ROS2 토픽을 통해 통신합니다.

---

## 1. 🎨 시스템 설계 및 플로우 차트

### 1-1. 시스템 설계도 (System Architecture)
<p align="center">
  <img src="./images/system_design.png" alt="시스템 설계도 이미지" width="400">
</p>

* *설명: 로봇 제어 PC와 Web Hub PC가 분리되어 구동되며, ROS2 토픽 통신으로 센서 데이터와 제어 신호를 주고받는 아키텍처입니다.*

### 1-2. 플로우 차트 (Flow Chart)
<p align="center">
  <img src="./images/flow_chart.png" alt="플로우 차트 이미지" width="300" height="300">
</p>

* *설명: [음성 주문(STT) ➡️ 박스 탐색 ➡️ 상품 탐색(ROI & 2-Step 검증) ➡️ 적응형 파지 ➡️ 충돌 회피 적재 ➡️ 박스 정렬(Shaking) ➡️ 교체] 로 이어지는 무한 루프 자동화 프로세스입니다.*

---

## 2. 🖥️ 시스템 구성 노드 (System Components)

### 🤖 1) 로봇/비전 PC 구성 (Robot/Vision)
* **메인 로봇 제어:** 2.5D 지형도를 분석하여 3D 공간 상의 최적 위치에 상품을 다층 적재 및 물리적 정렬(Shaking) 수행.
* **객체 탐지 (YOLO & OpenCV):** 타겟 상품의 좌표 및 Depth 추출.

### 🌐 2) Web PC Hub 구성 (웹 + 컨베이어 + STT)
* **Flask Web (`app.py`):** 고객용 POS 화면 및 주문 현황 폴링 API 제공.
* **Wakeword + STT 노드 (`ros_nodes/get_keyword.py`):** 음성 호출(Wakeword) 감지 후 STT를 통해 시스템 가동 명령 하달.
* **컨베이어 제어 노드 (`ros_nodes/belt_control_node.py`):** 아두이노(시리얼)와 연동하여 벨트 정지/가동 제어 및 타임아웃 상태 모니터링.

---

## 3. 📡 토픽 통신 계약 및 클래스 매핑 (ROS2 Communication)
각 시스템 간 통신을 위해 정의된 필수 ROS2 Topic 및 클래스 규칙입니다.

### 3-1. 클래스 매핑 (YOLO ID)
| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Class** | pringles | pepsi | tuna | cube | gum | spam | box |

### 3-2. 토픽 계약 (Topic Contracts)
* **로봇/YOLO PC ➡️ Web Hub PC**
  * `/product_detection` `(std_msgs/Int64)`: 로봇이 상품을 파지했을 때 해당 클래스 ID 전송 (장바구니 갱신용).
  * `/box_status` `(std_msgs/String)`: 박스 만석 시 `"FULL"`, 교체 완료 시 `"READY"` 전송.
* **Web Hub PC 내부 및 로봇 방향 송신**
  * `/payment_status` `(std_msgs/String)`: STT 노드에서 결제 시작 인식 시 `"payment start"` 발행 (로봇 가동 트리거).
  * `/waiting_for_payment` `(std_msgs/String)`: 컨베이어가 TIMEOUT_STOP 상태일 때 `"ready_for_payment"` 발행.

---

## 4. 🛠️ 운영체제, 하드웨어 및 의존성

* **OS:** Ubuntu 22.04 LTS
* **ROS Version:** ROS 2 Humble
* **Language:** Python 3.10
* **IDE:** VS Code

### 하드웨어 장비 (Hardware)
| 장비명 (Model) | 수량 | 비고 |
|:---:|:---:|:---|
| Doosan Robot (m0609) | 1 | 6축 협동 로봇 (메인 매니퓰레이터) |
| OnRobot RG2 | 1 | 로봇 그리퍼 (Modbus TCP 통신) |
| Intel RealSense Depth Camera | 1 | 3D 비전 및 2.5D 지형도 맵핑용 |
| 컨베이어 벨트 & IR 센서 | 1 | 상품 이송 및 정지 감지용 |
| Arduino Uno | 1 | 컨베이어 벨트 시리얼 제어용 |

### 의존성 (Dependencies)
pip install opencv-python numpy scipy ultralytics pymodbus pyserial flask


5. ▶️ 실행 순서 (Usage Guide)
시스템이 두 PC로 나뉘어 구동되므로, 각각의 장치에서 아래 순서대로 실행합니다.

💻 A. 로봇/비전 PC 실행
# Step 1. 로봇 및 카메라 초기화
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=192.168.137.100 port:=12345 model:=m0609

ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# Step 2. 비전 탐지 노드 실행
ros2 run pick_test detection_opencv

# Step 3. 로봇 메인 제어 노드 실행
ros2 run pick_test robot_move_wh

💻 B. Web Hub PC 실행 (터미널 3개 필요)
# Step 1. 컨베이어 벨트 노드 실행 (아두이노 시리얼 통신)
ros2 run pick_test belt_control_node

# Step 2. 음성 인식(STT) 노드 실행
python3 src/pick_test/ros_nodes/get_keyword.py

# Step 3. 웹 서버(POS) 실행
python3 src/pick_test/app.py
