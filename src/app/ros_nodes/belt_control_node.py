import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time


class BeltManagerNode(Node):
    def __init__(self):
        super().__init__('belt_manager_node')

        self.port = '/dev/ttyACM0'
        try:
            self.ser = serial.Serial(self.port, 9600, timeout=0.01)
            self.get_logger().info(f"✅ 아두이노 연결 성공: {self.port}")
            time.sleep(2)
        except Exception as e:
            self.get_logger().error(f"❌ 아두이노 연결 실패: {e}")
            exit()

        # 세션 시작 시 벨트 시작 + waiting_for_payment one-shot latch reset
        self.subscription = self.create_subscription(
            String,
            '/payment_status',
            self.payment_callback,
            10,
        )

        self.publisher = self.create_publisher(
            String,
            '/waiting_for_payment',
            10,
        )

        self.timer = self.create_timer(0.1, self.check_arduino_status)

        self.waiting_payment_sent = False

        self.get_logger().info("🎤 STT '계산' 명령 대기 중...")

    def payment_callback(self, msg):
        if msg.data == "payment start":
            self.get_logger().warn("💳 [명령] 계산 의도 확인됨. 상품 탐색 모드 시작!")
            self.waiting_payment_sent = False
            self.ser.write(b'1')

    def check_arduino_status(self):
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line == "STOPPED_BY_IR":
                    self.get_logger().info("📦 [상태] 상품이 센서에 감지되었습니다. 벨트 정지.")
                elif line == "TIMEOUT_STOP":
                    self.get_logger().error("🔔 [상태] 10초간 상품이 없어 시스템이 자동 종료되었습니다. 결제 안내를 시작합니다.")

                    if not self.waiting_payment_sent:
                        pub_msg = String()
                        pub_msg.data = "ready_for_payment"
                        self.publisher.publish(pub_msg)
                        self.waiting_payment_sent = True
                        self.get_logger().info(f"📤 '/waiting_for_payment' 토픽 1회 퍼블리시 완료: {pub_msg.data}")
                    else:
                        self.get_logger().info("⏭️ '/waiting_for_payment' 는 이미 보냈으므로 재발행하지 않음")

                elif line == "SYSTEM_START":
                    self.get_logger().info("🏃 [상태] 첫 상품 탐색을 시작합니다 (무제한 대기).")
            except Exception:
                pass

    def stop_system(self):
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
