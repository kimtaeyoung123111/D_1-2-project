"""
================================================================================
[로봇 비전 다층 적재 시스템 (3D Pick-and-Place) 핵심 요약]

1. 전체적인 역할 요약:
이 노드는 RealSense 뎁스 카메라와 YOLO 비전 시스템을 활용하여, 두산 로봇(m0609)이 
컨베이어 벨트 위의 상품을 인식하고, 박스 내부의 3D 공간(2.5D 지형도)을 계산하여 
최적의 위치에 다층(Multi-layer)으로 차곡차곡 적재하는 '로봇의 메인 두뇌(Control Tower)' 역할을 합니다.
특히 비전 오차 보정, 스팸과 같은 비대칭 상품의 90도 회전 적재, 그리고 카메라 사각지대에 
가려진 1층 물건을 기억하는 '가상 기억 장치' 등 고도화된 공간 지각 알고리즘이 탑재되어 있습니다.

2. ROS 2 통신 (Topic & Service) 구조:
- [Publisher] /product_detection (Int64): 
  로봇이 상품을 집어 올렸을 때, 해당 상품의 ID 번호를 발행하여 장바구니 DB를 업데이트합니다.
- [Publisher] /box_status (String): 
  박스에 빈 공간이 없어 꽉 찼다고 판단되면 "FULL" 메시지를 발행하여 AGV나 작업자를 호출합니다.
- [Subscriber] /box_status (String): 
  작업자가 새 박스로 교체한 뒤 "READY" 메시지를 보내면, 이를 듣고 대기 상태를 해제하여 작업을 재개합니다.
- [Subscriber] /payment_status (String): 
  결제 시스템에서 "payment start" 메시지를 받으면 시스템 대기를 해제하고 로봇을 가동합니다.
- [Client] /get_3d_position (SrvDepthPosition): 
  비전 노드에게 "박스 찾아줘(box)", "박스 안쪽 물건 찾아줘(box_contents)" 등의 
  요청을 보내고, 물체의 X, Y, Z 깊이값과 크기, 클래스 ID를 응답받는 가장 핵심적인 서비스 통신입니다.
================================================================================
"""

import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init
import cv2
import math
from std_msgs.msg import String, Int64

from od_msg.srv import SrvDepthPosition
from ament_index_python.packages import get_package_share_directory
from pick_test.onrobot import RG

# 로봇 및 그리퍼 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("rokey_simple_move", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, movejx, get_current_posx, mwait, trans
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

from DSR_ROBOT2 import set_tool, set_tcp, ROBOT_MODE_MANUAL, ROBOT_MODE_AUTONOMOUS
from DSR_ROBOT2 import set_robot_mode

########### Gripper Setup ############
GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)

# Tool과 TCP 설정시 매뉴얼 모드로 변경해서 진행
set_robot_mode(ROBOT_MODE_MANUAL)
set_tool(ROBOT_TOOL)
set_tcp(ROBOT_TCP)

set_robot_mode(ROBOT_MODE_AUTONOMOUS)
time.sleep(2)  # 설정 안정화를 위해 잠시 대기

########### Robot Controller ############

class RobotController(Node):
    def __init__(self):
        super().__init__("pick_and_place", namespace=ROBOT_ID)

        # --- 적재 관리용 변수 ---
        self.conveyor_z = 75.15     
        self.box_width = 266.0      
        self.box_length = 185.0     
        self.grid_res = 5           

        self.box_origin = None      
        self.box_angle = 0.0        
        self.is_at_home = False     
        
        self.init_robot()           
        
        self.item_publisher = self.create_publisher(Int64, '/product_detection', 10)

        self.depth_client = self.create_client(SrvDepthPosition, "/get_3d_position")
        while not self.depth_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for depth position service...")
        
        self.box_status_pub = self.create_publisher(String, '/box_status', 10)
        self.box_status_sub = self.create_subscription(String, '/box_status', self.box_status_callback, 10)
        
        self.is_waiting_for_box = False
        self.has_shaken_box = False
        self.hidden_memory = []

        self.depth_request = SrvDepthPosition.Request()

        # ==========================================================
        # 🌟 [적용 완료] 1. 초기 기본 규격표 세팅 (여유로운 사이즈)
        # ==========================================================
        self.REAL_SIZES = {
            0: (75.0, 75.0),   # 프링글스
            1: (75.0, 75.0),   # 펩시
            2: (85.0, 85.0),   # 참치캔
            3: (60.0, 60.0),   # 큐브
            4: (70.0, 70.0),   # 껌통
            5: (60.0, 105.0)   # 스팸
        }

        # ==========================================================
        # 🌟 [적용 완료] 2. 결제 시스템 연동 (구독자 및 상태 변수)
        # ==========================================================
        self.payment_status_sub = self.create_subscription(String, '/payment_status', self.payment_status_callback, 10)
        self.is_payment_completed = False  

        # 로봇 통제 함수 실행
        self.robot_control()
    
    def box_status_callback(self, msg):
        if msg.data == "READY":
            self.get_logger().info("✅ [수신] 새로운 박스 투입 확인! 대기 상태를 해제합니다.")
            self.is_waiting_for_box = False
            self.hidden_memory = []  
            self.has_shaken_box = False

            # ==========================================================
            # 🌟 [적용 완료] 3. 새 박스 투입 시 평소 규격으로 다시 복구!
            # ==========================================================
            self.REAL_SIZES = {
                0: (75.0, 75.0),   
                1: (75.0, 75.0),   
                2: (85.0, 85.0),   
                3: (60.0, 60.0),   
                4: (70.0, 70.0),   
                5: (60.0, 105.0)   
            }
            self.get_logger().info("📐 새 박스 투입: 넉넉한 탐색 규격으로 리셋합니다.")

    # ==========================================================
    # 🌟 [적용 완료] 4. 결제 완료 콜백 함수
    # ==========================================================
    def payment_status_callback(self, msg):
        if msg.data == "payment start":
            self.get_logger().info("💳 [결제 완료] 'payment start' 신호 수신! 로봇을 가동합니다.")
            self.is_payment_completed = True

    # 🛠️ [3D 입체 적재 알고리즘 적용] [2.5D 지도로 확인]
    def find_empty_space(self, items_in_box, raw_w, raw_h, class_id=-1):
        """[알고리즘 4-6번] 2.5D 지형도(Height Map)를 이용한 3D 공간 탐색"""

        # 박스 길이에 따른 지도 칸 나누기 
        rows = int(self.box_length // self.grid_res)
        cols = int(self.box_width // self.grid_res)
        
        grid = np.full((rows, cols), self.box_floor_z, dtype=float)

        wall_pad_mm = 20.0  
        wp_lr = int(wall_pad_mm // self.grid_res)
        wp_tb = int((wall_pad_mm / 4) // self.grid_res) 

        if wp_lr > 0 or wp_tb > 0:
            if wp_tb > 0: grid[0:wp_tb, :] = 9999.0       
            if wp_tb > 0: grid[-wp_tb:, :] = 9999.0       
            if wp_lr > 0: grid[:, 0:wp_lr] = 9999.0       

        if class_id == 1: current_margin = 0.0
        elif class_id == 2: current_margin = 22.0
        elif class_id == 3: current_margin = 12.0
        elif class_id == 4: current_margin = 8.0
        elif class_id == 5: current_margin = 10.0
        else: current_margin = 7.0

        if not hasattr(self, 'aligned_T') or self.aligned_T is None:
            self.get_logger().error("aligned_T가 없습니다.")
            return None

        inv_T = np.linalg.inv(self.aligned_T)
        w_half = self.box_width / 2
        l_half = self.box_length / 2

        all_items = items_in_box + getattr(self, 'hidden_memory', [])

        for item in all_items:
            if len(item) < 6: continue
            
            item_local = np.dot(inv_T, [item[0], item[1], item[2], 1])
            lx = max(0.0, item_local[0] + w_half)   
            ly = max(0.0, item_local[1] + l_half)   

            box_item_id = int(item[6]) if len(item) >= 7 else -1
            
            # ==========================================================
            # 🌟 [적용 완료] 5. 자체 선언된 REAL_SIZES 대신 클래스 공통 규격 사용
            # ==========================================================
            if box_item_id in self.REAL_SIZES:
                item_w = self.REAL_SIZES[box_item_id][0]
                item_h = self.REAL_SIZES[box_item_id][1]
            else:
                item_w = max(item[4], item[5])
                item_h = min(item[4], item[5])

            item_top_z = item[2]
            c_s = max(0, int((lx - item_w/2) // self.grid_res))
            r_s = max(0, int((ly - item_h/2) // self.grid_res))
            c_e = min(cols, c_s + int(item_w // self.grid_res) + 1)
            r_e = min(rows, r_s + int(item_h // self.grid_res) + 1)
            
            grid[r_s:r_e, c_s:c_e] = np.maximum(grid[r_s:r_e, c_s:c_e], item_top_z)

        # ---------------- 박스에 상품 적재 --------------------
        if class_id in self.REAL_SIZES:
            new_w = self.REAL_SIZES[class_id][0]
            new_h = self.REAL_SIZES[class_id][1]
        else:
            new_w = max(raw_w, raw_h)
            new_h = min(raw_w, raw_h)
        
        margin_w = current_margin + 2.0 
        margin_h = 3.0                  

        orientations = [
            {'w': new_w, 'h': new_h, 'mw': margin_w, 'mh': margin_h, 'rz_offset': 0.0}
        ]
        
        if class_id == 5:
            orientations.append(
                {'w': new_h, 'h': new_w, 'mw': margin_h, 'mh': margin_w, 'rz_offset': 90.0}
            )

        for orient in orientations:
            curr_w, curr_h = orient['w'], orient['h']
            curr_mw, curr_mh = orient['mw'], orient['mh']
            curr_rz_offset = orient['rz_offset']

            nw = int((curr_w + curr_mw) // self.grid_res)
            nh = int((curr_h + curr_mh) // self.grid_res)

            best_z = 9999.0
            best_r, best_c = -1, -1
            best_tx, best_ty, best_rz = 0, 0, 0

            for r in range(rows - nh):
                for c in range(cols - nw):
                    region = grid[r : r + nh, c : c + nw]
                    max_z = np.max(region) 
                    
                    gap_mask = region < (max_z - 15.0)
                    gap_area_ratio = np.sum(gap_mask) / (nw * nh) 
                    
                    if gap_area_ratio > 0.20: continue
                    if max_z > self.box_floor_z + 90.0: continue
                        
                    if max_z < best_z:
                        best_z = max_z
                        best_r = r
                        best_c = c
                        
                        local_target_x = (c * self.grid_res) + (curr_w / 2) + (curr_mw / 2)
                        local_target_y = (r * self.grid_res) + (curr_h / 2) + (curr_mh / 2)
                        local_from_center_x = -w_half + local_target_x
                        local_from_center_y = -l_half + local_target_y
                        
                        target_base = np.dot(self.aligned_T, [local_from_center_x, local_from_center_y, 0, 1])
                        best_tx, best_ty = target_base[0], target_base[1]
                        
                        best_rz = self.box_origin[5] + curr_rz_offset

                        if class_id in [0, 1, 2, 4]:
                            self.get_logger().info("🌪️ 원통형 상품 감지! 각도를 45도 비틀어서 적재합니다!")
                            best_rz += 45.0

            if best_r != -1:
                if curr_rz_offset == 90.0:
                    self.get_logger().info("🔄 스팸 정방향 공간 부족! 90도 회전하여 빈 공간에 꽂아 넣습니다.")
                break  

        norm_grid = np.clip((grid - self.box_floor_z) / 100.0 * 255, 0, 255).astype(np.uint8)
        grid_resized = cv2.resize(norm_grid, (810, 555), interpolation=cv2.INTER_NEAREST)
        grid_color = cv2.applyColorMap(grid_resized, cv2.COLORMAP_JET)

        if best_r != -1: 
            scale_x, scale_y = 810 / cols, 555 / rows
            px_start = (int(best_c * scale_x), int(best_r * scale_y))
            px_end = (int((best_c + nw) * scale_x), int((best_r + nh) * scale_y))
            
            cv2.rectangle(grid_color, px_start, px_end, (0, 255, 0), 2)
            cv2.putText(grid_color, f"STACK Z:{int(best_z)}", (px_start[0], px_start[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("2.5D Elevation Map (Live)", cv2.flip(grid_color, -1))
        cv2.waitKey(2000 if best_r != -1 else 1)

        if best_r != -1:
            return [best_tx, best_ty, best_z, self.box_origin[3], self.box_origin[4], best_rz]
        
        return None

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path, allow_pickle=True)
        coord = np.append(np.array(camera_coords), 1)
        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    # 로봇 동작 함수
    def robot_control(self):
        self.get_logger().info("⏳ 시스템 대기 중... 결제 완료 신호를 기다립니다. (/payment_status)")
        
        # ==========================================================
        # 🌟 [적용 완료] 6. 결제 완료 전까지 무한 대기 (스위치 역할)
        # ==========================================================
        while rclpy.ok() and not self.is_payment_completed:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("🚀 결제 확인 완료! 자동 Pick-and-Place 모드를 시작합니다.")
        gripper2cam_path = "/home/kim/Tutorial_2026/Tutorial/Calibration_Tutorial/T_gripper2camera.npy"
        
        while rclpy.ok():
            self.get_logger().info("📦 [1] 박스의 최신 위치 및 내부 공간을 파악합니다 (빈 손).")
            
            self.align_to_box_first()
            if self.box_origin is None:
                self.get_logger().warn("⚠️ 박스를 찾을 수 없어 대기합니다.")
                time.sleep(1.0)
                continue
            
            self.get_logger().info("📸 카메라 안정화 및 정밀 탐지를 위해 2초간 대기합니다...")
            time.sleep(2.0)

            self.depth_request.target = "box_contents"
            future = self.depth_client.call_async(self.depth_request)
            rclpy.spin_until_future_complete(self, future)

            items_in_box = [] 
            if future.result():
                res = future.result().depth_position.tolist()
                if res and len(res) >= 1:
                    num_items = int(res[0])
                    curr_px = get_current_posx()[0]

                    for i in range(num_items):
                        start_idx = 1 + (i * 7)
                        end_idx = start_idx + 7
                        if len(res) >= end_idx:
                            item_data = res[start_idx:end_idx]
                            base_coords = self.transform_to_base(item_data[:3], gripper2cam_path, curr_px)
                            item_data[0:3] = base_coords[0:3]
                            items_in_box.append(item_data)

            self.is_at_home = False
            self.init_robot() 
            time.sleep(1.0)

            target_pos = None 
            current_obj_size = [0.0, 0.0]

            self.get_logger().info("🔍 [2] 상품 탐색 중...")
            while rclpy.ok():
                self.depth_request.target = ""
                depth_future = self.depth_client.call_async(self.depth_request)
                rclpy.spin_until_future_complete(self, depth_future)

                if depth_future.result():
                    result = depth_future.result().depth_position.tolist()
                    if len(result) >= 6 and sum(result[:3]) != 0:
                        camera_coords = result[:3]
                        current_obj_size = [result[4], result[5]]
                        
                        class_id = -1
                        if len(result) >= 7:
                            class_id = int(result[6])
                        
                        robot_posx = get_current_posx()[0]
                        td_coord = self.transform_to_base(camera_coords, gripper2cam_path, robot_posx)
                        
                        if class_id == 1:   
                            td_coord[2] += -38.0
                            self.get_logger().info("🥤 펩시 탐지됨! 파지 깊이를 -40으로 조절합니다.")
                        elif class_id == 2: 
                            td_coord[2] += -55.0
                        else:
                            td_coord[2] += -50.0 
                            
                        td_coord[2] = max(td_coord[2], 2.0)

                        final_angle = result[3]
                        if current_obj_size[0] < current_obj_size[1]:
                            final_angle += 90.0
                        
                        new_rz = robot_posx[5] + final_angle - 90
                        target_pos = [td_coord[0], td_coord[1], td_coord[2], robot_posx[3], robot_posx[4], new_rz]
                        
                        self.get_logger().info(f"✅ 상품 발견! 크기: {current_obj_size[0]}x{current_obj_size[1]}")
                        break 
                time.sleep(1.0) 

            self.get_logger().info("🧠 [3] 파악된 빈 공간에 물건이 들어갈 수 있는지 계산합니다.")
            drop_pose = self.find_empty_space(items_in_box, current_obj_size[0], current_obj_size[1], class_id)

            if drop_pose is None:
                if not self.has_shaken_box:
                    self.get_logger().warn("⚠️ 박스 만석 감지! 1차 조치로 박스를 흔들어 공간을 확보해 봅니다.")
                    self.shake_box()            
                    self.has_shaken_box = True  
                    
                    self.box_origin = None
                    self.last_known_angle = None
                    continue  
                else:
                    self.get_logger().error("🛑 흔들어도 공간이 없습니다! 찐 만석! 박스 교체를 요청합니다.")
                    
                    msg = String()
                    msg.data = "FULL"
                    self.box_status_pub.publish(msg)
                    self.get_logger().info("📢 새 박스 요청 신호(/box_status: 'FULL')를 발송했습니다.")
                    
                    self.is_waiting_for_box = True
                    self.get_logger().info("⏳ 작업자/AGV의 완료 신호(/box_status: 'READY')를 무한 대기합니다...")
                    
                    while rclpy.ok() and self.is_waiting_for_box:
                        rclpy.spin_once(self, timeout_sec=0.1)
                        
                    self.box_origin = None
                    self.last_known_angle = None
                    continue

            self.get_logger().info("✅ 적재 위치 확인 완료. 파지를 시작합니다.")
            self.pick_object(target_pos, class_id)

            self.place_object(drop_pose, target_pos[2])

            # ==========================================================
            # 🌟 [적용 완료] 7. 성공 시 흔들기 찬스만 충전 (규격은 유지!)
            # ==========================================================
            if self.has_shaken_box:
                self.has_shaken_box = False
                self.get_logger().info("🔋 흔들기 후 빈 공간 적재 성공! 흔들기 찬스를 다시 1회 충전합니다.")

            target_floor_z = drop_pose[2] 
            if target_floor_z > self.box_floor_z + 10.0:
                closest_item = None
                min_dist = 9999.0
                for item in items_in_box:
                    dist = math.hypot(item[0] - drop_pose[0], item[1] - drop_pose[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_item = item
                
                if closest_item is not None and min_dist < 60.0:
                    self.hidden_memory.append(closest_item) 
                    self.get_logger().info("🧠 [기억 장치] 2층 적재 감지! 가려진 1층 상품의 크기와 위치를 기억합니다.")

    # ==========================================================
    # 🌟 [업그레이드] 박스 정렬(흔들기) 전용 함수 (Pull -> Tilt -> Shake -> Push)
    # ==========================================================
    def shake_box(self):
        """박스를 잡고 안으로 끌고 온 뒤, 기울여서 흔들고 다시 제자리에 밀어놓습니다."""
        self.get_logger().info("📳 빈 공간 확보를 위해 박스 당겨서 흔들기를 시작합니다!")

        outward_margin = 0  
        side_margin = 0.0

        local_grab_x = (self.box_width / 2.0) + outward_margin  
        local_grab_y = side_margin                
        target_base = np.dot(self.aligned_T, [local_grab_x, local_grab_y, 0, 1])

        grab_z = self.box_floor_z + 80.0
        grab_rz = self.box_origin[5]
        grab_pose = [target_base[0], target_base[1], grab_z, self.box_origin[3], self.box_origin[4], grab_rz]

        approach_pose = list(grab_pose)
        approach_pose[2] += 100.0

        gripper.move_gripper(550)
        time.sleep(1.0) 
        movel(approach_pose, vel=200, acc=100); mwait()
        movel(grab_pose, vel=100, acc=50); mwait()
        
        gripper.close_gripper()
        while gripper.get_status()[0]: time.sleep(0.5)

        pull_distance = 150.0 
        
        pulled_local_x = local_grab_x - pull_distance
        pulled_local_y = local_grab_y
        
        pulled_base = np.dot(self.aligned_T, [pulled_local_x, pulled_local_y, 0, 1])
        
        pull_pose = list(grab_pose)
        pull_pose[0] = pulled_base[0]
        pull_pose[1] = pulled_base[1]
        
        self.get_logger().info(f"⬅️ 관절 한계(Singularity) 방지를 위해 박스를 {pull_distance}mm 당깁니다.")
        movel(pull_pose, vel=150, acc=50); mwait()

        # ==========================================================
        # 🌟 [적용 완료] 8. 순서 꼬임(UnboundLocalError) 완벽 해결
        # ==========================================================
        lift_pose = list(pull_pose)
        
        pull_pose_up = list(pull_pose)
        pull_pose_up[2] += 80.0

        # 여기서 위로 들어올리는 동작 실행 (주석 해제/유지 여부는 상황에 맞게!)
        movel(pull_pose_up, vel=200, acc=50); mwait()

        self.get_logger().info("🌪️ 쉐킷 쉐킷! 손목 스냅을 넣어 주기적으로 흔듭니다!")
        
        shake_offset = 30.0       
        shake_angle_offset = 8.0  
        
        shake_left_pos = np.dot(self.aligned_T, [pulled_local_x, pulled_local_y + shake_offset, 0, 1])
        shake_right_pos = np.dot(self.aligned_T, [pulled_local_x, pulled_local_y - shake_offset, 0, 1])

        pose_left = [shake_left_pos[0], shake_left_pos[1], lift_pose[2], lift_pose[3], lift_pose[4], grab_rz + shake_angle_offset]
        pose_right = [shake_right_pos[0], shake_right_pos[1], lift_pose[2], lift_pose[3], lift_pose[4], grab_rz - shake_angle_offset]

        # (필요시 아래 for문 주석 해제하여 흔들기 발동)
        # for _ in range(3):
        #     movel(pose_left, vel=500, acc=400); mwait()  
        #     movel(pose_right, vel=500, acc=400); mwait() 
        # movel(lift_pose, vel=200, acc=50); mwait()

        self.get_logger().info("⬇️ 바닥에 수평으로 내려놓습니다.")
        
        down_pose = list(pull_pose)
        movel(down_pose, vel=100, acc=50); mwait() 

        self.get_logger().info("➡️ 카메라가 다시 찾기 편하도록 박스를 원래 위치로 밀어 넣습니다.")
        movel(grab_pose, vel=150, acc=50); mwait()
        
        gripper.open_gripper()
        time.sleep(1.0)

        movel(approach_pose, vel=200, acc=100); mwait()
        self.get_logger().info("✅ 당겨서 흔들고 원위치 복귀 완료! 새로운 빈 공간을 찾습니다.")

        # ==========================================================
        # 🌟 [적용 완료] 9. 흔들기 완료 시 타이트한 규격으로 변경!
        # ==========================================================
        self.REAL_SIZES = {
            0: (70.0, 70.0),   
            1: (65.0, 65.0),   
            2: (80.0, 80.0),   
            3: (55.0, 55.0),   
            4: (60.0, 60.0),   
            5: (60.0, 105.0)   
        }
        self.get_logger().info("📐 흔들기 완료: 타이트한 탐색 규격으로 전환합니다.")

    def align_to_box_first(self):
        """루프 시작 시 박스 위치로 가서 각도를 갱신하고 반환 (박스가 삐뚤어졌는지 확인)"""
        self.get_logger().info("📦 [시퀀스 1] 박스 위치 실시간 탐색 및 갱신 시작")
        box_pos_joint = [13.67, 17.08, 43.32, 0.11, 119.6, 193.59]  # 절대 좌표로 가서 박스 탐지 준비
        movej(box_pos_joint, vel=VELOCITY, acc=ACC)
        mwait()
        
        # OpenCV로 box를 탐지했을 때
        self.depth_request.target = "box" 
        future = self.depth_client.call_async(self.depth_request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result():
            res = future.result().depth_position.tolist()
            if len(res) >= 6 and res[2] > 0: 
                curr_px = get_current_posx()[0]
                gripper2cam_path = "/home/kim/Tutorial_2026/Tutorial/Calibration_Tutorial/T_gripper2camera.npy"
                box_center_base = self.transform_to_base(res[:3], gripper2cam_path, curr_px)

                self.box_floor_z = 5.0
                box_w, box_h = res[4], res[5] 
                raw_angle = res[3]
                
                if box_w < box_h: raw_angle -= 90.0 
                
                if getattr(self, 'last_known_angle', None) is None:
                    while raw_angle > 90.0: raw_angle -= 180.0
                    while raw_angle <= -90.0: raw_angle += 180.0
                    final_box_angle = raw_angle
                else:
                    diff = raw_angle - self.last_known_angle
                    while diff > 90.0:
                        raw_angle -= 180.0
                        diff = raw_angle - self.last_known_angle
                    while diff <= -90.0:
                        raw_angle += 180.0
                        diff = raw_angle - self.last_known_angle
                    final_box_angle = raw_angle
                
                self.last_known_angle = final_box_angle
                new_rz = curr_px[5] + final_box_angle

                self.aligned_T = self.get_robot_pose_matrix(
                    box_center_base[0], box_center_base[1], box_center_base[2], 
                    curr_px[3], curr_px[4], new_rz
                )

                local_corner = np.array([-self.box_width / 2, -self.box_length / 2, 0, 1])
                base_corner = np.dot(self.aligned_T, local_corner)

                gripper2cam = np.load(gripper2cam_path, allow_pickle=True)
                local_cam_offset = np.array([gripper2cam[0, 3], gripper2cam[1, 3], gripper2cam[2, 3]])
                R_view = Rotation.from_euler("ZYZ", [curr_px[3], curr_px[4], new_rz], degrees=True).as_matrix()
                base_cam_offset = np.dot(R_view, local_cam_offset)
                
                tcp_target_x = box_center_base[0] - base_cam_offset[0]
                tcp_target_y = box_center_base[1] - base_cam_offset[1]
                
                new_box_view_pos = [tcp_target_x, tcp_target_y, 260.0, 
                                    curr_px[3], curr_px[4], new_rz]
                
                self.get_logger().info(f"🎯 카메라 중앙 정렬을 위해 TCP가 ({tcp_target_x:.1f}, {tcp_target_y:.1f})로 이동합니다.")
                movel(new_box_view_pos, vel=VELOCITY, acc=ACC); mwait()
                time.sleep(2)
                
                self.box_angle = final_box_angle
                self.box_origin = [base_corner[0], base_corner[1], 150, 
                                curr_px[3], curr_px[4], new_rz]
                
                return self.box_origin
            else:    
                self.get_logger().error("❌ 처음부터 박스를 못 찾았습니다.")
                return None

    def init_robot(self):
        if self.is_at_home: return
        initial_pose = [-33.37, 4.15, 50.59, 0.02, 125.27, 146.84]
        movej(initial_pose, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()
        self.is_at_home = True

    def pick_object(self, target_pos, class_id):
        """상품만 집어 올리는 함수 (파지)"""
        self.is_at_home = False
        self.get_logger().info("✅ 상품 픽업을 먼저 시작합니다.")
        
        approach_pos = list(target_pos); approach_pos[2] += 100.0 
        movel(approach_pos, vel=500, acc=300); mwait()
        movel(target_pos, vel=VELOCITY, acc=ACC); mwait()
        
        gripper.close_gripper()
        while gripper.get_status()[0]: time.sleep(0.5)

        target_pos_up = list(target_pos); target_pos_up[2] += 200.0 
        movel(target_pos_up, vel=500, acc=300); mwait()

        pub_msg = Int64()
        pub_msg.data = int(class_id) 
        self.item_publisher.publish(pub_msg) 
        self.get_logger().info(f"📤 상품이 장바구니에 담겼습니다! ID: {pub_msg.data}")

    def place_object(self, drop_pose, pick_z):
        """계산된 위치에 상품을 내려놓는 함수 (적재)"""
        target_floor_z = drop_pose[2] 

        if target_floor_z > self.box_floor_z + 10.0:
            drop_offset = 20.0  
            self.get_logger().info(f"🏢 2층 이상 적재 감지! Z 높이를 {drop_offset}mm 더 낮춰서 꾹 눌러줍니다.")
        else:
            drop_offset = 0.0  
            self.get_logger().info("🏠 1층 바닥 적재 감지! 기본 높이로 내려놓습니다.")
        
        actual_drop_z = target_floor_z + (pick_z - self.conveyor_z) - drop_offset
        drop_pose[2] = actual_drop_z

        if drop_pose[5] > 180.0: drop_pose[5] -= 360.0
        elif drop_pose[5] < -180.0: drop_pose[5] += 360.0

        self.get_logger().info(f"📦 적재 목표 위치: X={drop_pose[0]:.1f}, Y={drop_pose[1]:.1f}")
        self.get_logger().info(f"👇 살포시 놓기 Z 높이 계산 완료: {actual_drop_z:.1f}mm")  

        box_approach = list(drop_pose)
        box_approach[2] += 150.0
        self.get_logger().info(f"⬇️ [1단계] 적재 준비 높이(Z+150)로 수직 하강 중...")
        movel(box_approach, vel=500, acc=300); mwait()

        self.get_logger().info("⏬ [2단계] 살포시 적재 중...")
        movel(drop_pose, vel=100, acc=30); mwait()
        time.sleep(0.3)
        
        current_width = gripper.get_width()
        gripper.move_gripper(int(current_width)*10 + 50)
        time.sleep(1.0) 

        exit_pos = list(drop_pose)
        exit_pos[2] += 150.0 
        movel(exit_pos, vel=500, acc=300); mwait()
        self.get_logger().info("✅ 적재 완료 및 안전 구역 복귀")

def main(args=None):
    node = RobotController()
    while rclpy.ok():
        node.robot_control()
    rclpy.shutdown()
    node.destroy_node()

if __name__ == "__main__":
    main()