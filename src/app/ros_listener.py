import threading
import queue
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Int64

ID2NAME = {
    0: "pringles",
    1: "pepsi",
    2: "tuna",
    3: "cube",
    4: "gum",
    5: "spam",
}

# Queues consumed by Flask
product_queue: "queue.Queue[str]" = queue.Queue()
box_status_queue: "queue.Queue[str]" = queue.Queue()
waiting_payment_queue: "queue.Queue[str]" = queue.Queue()
payment_status_queue: "queue.Queue[str]" = queue.Queue()
voice_yesno_queue: "queue.Queue[str]" = queue.Queue()


class WebHubROS(Node):
    def __init__(self):
        super().__init__("web_hub_ros")

        # Subscriptions
        self.create_subscription(Int64, "product_detection", self._on_product, 10)
        self.create_subscription(String, "/box_status", self._on_box_status, 10)
        self.create_subscription(String, "/waiting_for_payment", self._on_waiting_for_payment, 10)
        self.create_subscription(String, "/payment_status", self._on_payment_status, 10)
        self.create_subscription(String, "/voice_yesno", self._on_voice_yesno, 10)

        # Optional publishers (web -> ROS)
        prompt_qos = QoSProfile(depth=1)
        prompt_qos.reliability = ReliabilityPolicy.RELIABLE
        prompt_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.session_start_pub = self.create_publisher(String, "/session_start", 10)
        self.voice_prompt_pub = self.create_publisher(String, "/voice_prompt_state", prompt_qos)
        self.box_status_pub = self.create_publisher(String, "/box_status", 10)

    def _on_product(self, msg: Int64):
        cls_id = int(msg.data)
        name = ID2NAME.get(cls_id, f"unknown_{cls_id}")
        product_queue.put(name)

    def _on_box_status(self, msg: String):
        box_status_queue.put(msg.data.strip())

    def _on_waiting_for_payment(self, msg: String):
        waiting_payment_queue.put(msg.data.strip())

    def _on_payment_status(self, msg: String):
        payment_status_queue.put(msg.data.strip())

    def _on_voice_yesno(self, msg: String):
        voice_yesno_queue.put(msg.data.strip().lower())

    def publish_voice_prompt(self, stage: str):
        msg = String()
        msg.data = (stage or "idle").strip().lower()
        self.voice_prompt_pub.publish(msg)

    def publish_box_status(self, status: str):
        msg = String()
        msg.data = (status or "").strip().upper()
        self.box_status_pub.publish(msg)


def start_ros_spin_in_thread():
    """Start rclpy spin in a daemon thread and return the node."""
    rclpy.init(args=None)
    node = WebHubROS()

    def _spin():
        try:
            rclpy.spin(node)
        except Exception:
            pass

    th = threading.Thread(target=_spin, daemon=True)
    th.start()
    return node
