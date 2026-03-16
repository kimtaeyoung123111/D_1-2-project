import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time

class BeltManagerNode(Node):
    def __init__(self):
        super().__init__('belt_manager_node')
        
        # 1. 시리얼 설정 (포트 번호 확인: /dev/ttyACM0)
        self.port = '/dev/ttyACM0' 
        try:
            self.ser = serial.Serial(self.port, 9600, timeout=0.01)
            self.get_logger().info(f"✅ 아두이노 연결 성공: {self.port}")
            time.sleep(2) # 연결 초기화 대기
        except Exception as e:
            self.get_logger().error(f"❌ 아두이노 연결 실패: {e}")
            exit()

        # 2. STT 결과 토픽 구독
        self.subscription = self.create_subscription(
            String,
            '/payment_status',
            self.payment_callback,
            10)
        
        # 2_1. 장바구니 등록 완료시 결제 안내 토픽 퍼블리쉬
        self.publisher = self.create_publisher(
            String,
            '/waiting_for_payment',
            10)

        # 3. 아두이노 상태 모니터링 타이머 (0.1초마다 시리얼 읽기)
        self.timer = self.create_timer(0.1, self.check_arduino_status)

        self.get_logger().info("🎤 STT '계산' 명령 대기 중...")

    def payment_callback(self, msg):
        """STT 토픽 수신 시 실행"""
        # "계산" 혹은 "payment start" 메시지가 오면 가동
        if msg.data == "payment start":
            self.get_logger().warn("💳 [명령] 계산 의도 확인됨. 상품 탐색 모드 시작!")
            self.ser.write(b'1') # 아두이노에 '1' 전송 (자동화 모드 ON)

    def check_arduino_status(self):
        """아두이노가 보내는 상태 메시지 처리"""
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line == "STOPPED_BY_IR":
                    self.get_logger().info("📦 [상태] 상품이 센서에 감지되었습니다. 벨트 정지.")
                elif line == "TIMEOUT_STOP":
                    self.get_logger().error("🔔 [상태] 10초간 상품이 없어 시스템이 자동 종료되었습니다. 결제 안내를 시작합니다.")
                    
                    # ==========================================
                    # 추가된 부분: 'waiting for payment' 토픽 발행
                    # ==========================================
                    pub_msg = String()
                    pub_msg.data = "ready_for_payment" # 필요에 따라 보낼 메시지 내용을 변경하세요.
                    self.publisher.publish(pub_msg)
                    self.get_logger().info(f"📤 'waiting for payment' 토픽 퍼블리시 완료: {pub_msg.data}")
                    # ==========================================

                elif line == "SYSTEM_START":
                    self.get_logger().info("🏃 [상태] 첫 상품 탐색을 시작합니다 (무제한 대기).")
            except Exception as e:
                pass

    def stop_system(self):
        """노드 종료 시 벨트 강제 정지"""
        if self.ser and self.ser.is_open:
            self.ser.write(b'0')
        self.get_logger().info("🛑 시스템 정지 및 포트 닫기.")

def main(args=None):
    rclpy.init(args=args)
    node = BeltManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_system()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
