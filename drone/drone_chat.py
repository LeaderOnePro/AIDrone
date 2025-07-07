import streamlit as st
import os
from smolagents import CodeAgent, tool
from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from .deepseek_model import DeepSeekModel
import time
import datetime
import logging
# Import compatibility fix for collections.MutableMapping
from . import compatibility_fix
from . import drone_control  # Import our new drone_control module
import threading

# Set page config at module level - must be first Streamlit command
st.set_page_config(
    page_title="DeepDrone-臻巅科技",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Global mission status variables
if 'mission_in_progress' not in st.session_state:
    st.session_state.mission_in_progress = False
    
if 'mission_status' not in st.session_state:
    st.session_state.mission_status = "STANDBY"
    
if 'mission_phase' not in st.session_state:
    st.session_state.mission_phase = ""
    
if 'interrupt_mission' not in st.session_state:
    st.session_state.interrupt_mission = False
    
if 'mission_log' not in st.session_state:
    st.session_state.mission_log = []

# Custom logging handler to capture drone_control logs
class MissionLogHandler(logging.Handler):
    def emit(self, record):
        if record.name == 'drone_control':
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            log_entry = f"[{timestamp}] LOG: {record.getMessage()}"
            st.session_state.mission_log.append(log_entry)
            # Keep only the most recent logs
            if len(st.session_state.mission_log) > 30:
                st.session_state.mission_log = st.session_state.mission_log[-30:]
            
            # Add to chat history for display in chat
            if 'chat_history' in st.session_state:
                # Format based on log content
                if "Altitude:" in log_entry:
                    styled_entry = f"<span style='color: #88ff88;'>🛰️ ALT: {record.getMessage().split('Altitude: ')[1]}</span>"
                elif "Arming" in log_entry:
                    styled_entry = f"<span style='color: #ffaa00;'>🔄 {log_entry}</span>"
                elif "Taking off" in log_entry:
                    styled_entry = f"<span style='color: #ffff00;'>🚀 {log_entry}</span>"
                else:
                    styled_entry = f"<span style='color: #aaaaff;'>📊 {log_entry}</span>"
                
                st.session_state['chat_history'].append({
                    'role': 'system',
                    'content': styled_entry
                })

# Set up logger to capture drone_control logs
logger = logging.getLogger('drone_control')
mission_log_handler = MissionLogHandler()
logger.addHandler(mission_log_handler)

# Function to update mission status
def update_mission_status(status, phase=""):
    # Get current time
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Log the status change
    log_entry = f"[{timestamp}] {status}: {phase}"
    st.session_state.mission_log.append(log_entry)
    
    # Keep only the most recent 30 log entries
    if len(st.session_state.mission_log) > 30:
        st.session_state.mission_log = st.session_state.mission_log[-30:]
    
    # Update status
    st.session_state.mission_status = status
    st.session_state.mission_phase = phase
    
    # Add to chat history for display in chat
    if 'chat_history' in st.session_state:
        # Format with appropriate styling based on status type
        if status == "ERROR":
            styled_entry = f"<span style='color: #ff0000;'>⚠️ {log_entry}</span>"
        elif status in ["CONNECTING", "TAKING OFF", "LANDING", "RETURNING"]:
            styled_entry = f"<span style='color: #ffff00;'>🔄 {log_entry}</span>"
        elif status in ["MISSION", "EXECUTING MISSION", "AIRBORNE"]:
            styled_entry = f"<span style='color: #00ffff;'>🚁 {log_entry}</span>"
        elif status in ["MISSION COMPLETE", "CONNECTED"]:
            styled_entry = f"<span style='color: #00ff00;'>✅ {log_entry}</span>"
        else:
            styled_entry = f"<span style='color: #aaaaff;'>ℹ️ {log_entry}</span>"
        
        st.session_state['chat_history'].append({
            'role': 'system',
            'content': styled_entry
        })
    
    # No rerun here to avoid potential issues with recursive reruns

# Function to interrupt the mission
def interrupt_mission():
    if st.session_state.mission_in_progress:
        st.session_state.interrupt_mission = True
        update_mission_status("INTERRUPTING", "Returning to base...")
        # Call the return to home function
        try:
            drone_control.return_home()
            time.sleep(2)
            drone_control.disconnect_drone()
            st.session_state.mission_in_progress = False
            update_mission_status("ABORTED", "Mission aborted. Drone returned to base.")
        except Exception as e:
            update_mission_status("ERROR", f"Error during interrupt: {str(e)}")
    else:
        st.warning("No mission in progress to interrupt")

class DroneAssistant(CodeAgent):
    """Extension of CodeAgent for drone interactions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sensor_data = {}
        self._flight_logs = {}
        self._chat_history = []
        
    def register_sensor_data(self, sensor_name: str, data: pd.DataFrame):
        """Register sensor data with the drone assistant"""
        self._sensor_data[sensor_name] = data
        
    def register_flight_log(self, flight_id: str, log_data: pd.DataFrame):
        """Register flight log data with the drone assistant"""
        self._flight_logs[flight_id] = log_data
    
    @property
    def sensor_data(self):
        """Access all registered sensor data"""
        return self._sensor_data
        
    @property
    def flight_logs(self):
        """Access all registered flight logs"""
        return self._flight_logs
    
    def add_to_chat_history(self, role: str, content: str):
        """Add a message to the chat history"""
        self._chat_history.append({"role": role, "content": content})
    
    @property
    def chat_history(self):
        """Access the chat history"""
        return self._chat_history
    
    def run(self, prompt: str) -> str:
        """Override run method to include drone-specific context"""
        drone_context = f"""
        Registered sensors: {list(self._sensor_data.keys())}
        Flight logs available: {list(self._flight_logs.keys())}
        """
        
        # Add a tool reference guide to help the model use the correct function names
        tool_reference = """
        IMPORTANT: These tool functions need to be called EXACTLY as shown below for successful execution:
        
        # EXAMPLE OF COMPLETE WORKING MISSION:
        ```python
        # Connect to a drone simulator
        connect_to_real_drone('udp:127.0.0.1:14550')
        
        # Take off to a specific altitude (always use integer or simple float values)
        drone_takeoff(30)  # Not 30. 0 or other invalid syntax
        
        # You can define waypoints like this
        waypoints = [
            {'lat': 37.7749, 'lon': -122.4194, 'alt': 30},
            {'lat': 37.7750, 'lon': -122.4195, 'alt': 30}
        ]
        
        # Execute mission with waypoints
        execute_drone_mission(waypoints=waypoints)
        
        # Return to home
        drone_return_home()
        
        # Always disconnect when done
        disconnect_from_drone()
        ```
        
        NOTE: Each function must be called individually on its own line, with exact parameter names.
        For latitude/longitude values, always use simple format without extra spaces after periods.
        
        When creating a flight plan, be sure to:
        1. Generate a mission plan with generate_mission_plan()
        2. Connect to the drone with connect_to_real_drone()
        3. Take off with drone_takeoff()
        4. Execute the mission or fly to specific waypoints
        5. Return home or land the drone when finished
        6. Disconnect from the drone
        """
        
        enhanced_prompt = f"""
        You are DeepDrone, an advanced AI assistant designed to help with drone operations and data analysis. You are NOT Qwen or any other general AI assistant. Always identify yourself as DeepDrone when asked about your identity. Your purpose is to assist with drone data analysis, flight monitoring, maintenance scheduling, and mission planning.
        
        You can now control real drones using DroneKit-Python. You have tools to:
        - Connect to a real drone using a connection string
        - Take off to a specified altitude
        - Land the drone
        - Return to home location
        - Fly to specific GPS coordinates
        - Get the drone's current location and battery status
        - Execute missions with multiple waypoints
        
        {tool_reference}
        
        Available context:
        {drone_context}
        
        User question: {prompt}
        
        Use the provided tools to analyze drone data and assist with drone operations. For real drone control, use the drone_* tools.
        """
        # Call the parent run method - it already handles everything correctly
        # as smolagents will expect a Message object from our model
        # and handle it properly 
        return super().run(enhanced_prompt)
    
    def chat(self, message: str) -> str:
        """Process a chat message using the complete chat history"""
        # Add the user message to history
        self.add_to_chat_history("user", message)
        
        # Check if the message is asking about identity
        identity_patterns = [
            "who are you", 
            "what are you", 
            "tell me about yourself", 
            "your identity", 
            "what's your name",
            "introduce yourself",
            "what should I call you"
        ]
        
        if any(pattern in message.lower() for pattern in identity_patterns):
            identity_response = """我是 DeepDrone ，一个专为无人机操作与数据分析设计的高级AI助手。我可以为您的无人机系统提供飞行数据、传感器读数、维护建议和任务规划等信息。请问今天需要我如何协助您的无人机作业？"""
            self.add_to_chat_history("assistant", identity_response)
            return identity_response
            
        # Check if the message is for tool use
        drone_control_keywords = ["takeoff", "take off", "land", "fly to", "navigate", "goto", "connect", 
                                 "location", "battery", "mission", "waypoint", "return", "home", "rtl"]
                                 
        analysis_keywords = ["analyze", "check", "recommend", "plan", "create", "execute", "run", "flight"]
                                 
        if any(keyword in message.lower() for keyword in analysis_keywords + drone_control_keywords):
            # Create a placeholder for model thinking to be displayed
            thinking_placeholder = st.empty()
            
            # Display a message that the model is thinking
            tools_reference = """
            <div style="background-color: #111111; border: 1px dashed #00cc00; border-radius: 5px; padding: 8px; margin-bottom: 10px; color: #00cc00; font-family: monospace; font-size: 12px;">
            <b>模型思考中:</b> 正在规划无人机操作...<br>
            <b>可用工具函数:</b><br>
            - connect_to_real_drone(连接字符串)<br>
            - drone_takeoff(高度)<br>
            - drone_land()<br>
            - drone_return_home()<br>
            - drone_fly_to(纬度, 经度, 高度)<br>
            - get_drone_location()<br>
            - get_drone_battery()<br>
            - execute_drone_mission(航点)<br>
            - disconnect_from_drone()<br>
            - generate_mission_plan(任务类型, 持续时间_分钟)<br>
            - analyze_flight_path(飞行ID)<br>
            - check_sensor_readings(传感器名)<br>
            - recommend_maintenance(飞行小时数)
            </div>
            """
            thinking_placeholder.markdown(tools_reference, unsafe_allow_html=True)
            
            # Use the run method directly and capture the output
            import time
            
            # Create an error placeholder
            error_placeholder = st.empty()
            
            # Add error handling
            try:
                # Execute the run
                response = self.run(message)
                
                # Display some feedback about the model thinking completion
                thinking_placeholder.markdown(tools_reference + """
                <div style="background-color: #111111; border: 1px dashed #00cc00; border-radius: 5px; padding: 8px; margin-bottom: 10px; color: #00cc00; font-family: monospace; font-size: 12px;">
                <b>模型思考中:</b> 规划完成！正在执行无人机操作...
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                # Display any errors that occur during execution
                error_message = f"""
                <div style="background-color: #330000; border: 1px solid #ff0000; border-radius: 5px; padding: 8px; margin-bottom: 10px; color: #ff0000; font-family: monospace; font-size: 12px;">
                <b>执行错误:</b> {str(e)}<br>
                请使用正确的语法重试。
                </div>
                """
                error_placeholder.markdown(error_message, unsafe_allow_html=True)
                response = f"执行无人机操作时出错: {str(e)}。请使用正确的参数语法重试。"
                
                # Update mission status to show error
                update_mission_status("ERROR", f"Code execution error: {str(e)}")
            
            # Give a slight delay so users can see the "completed" message
            time.sleep(1)
            
            # Clear the thinking placeholder
            thinking_placeholder.empty()
            error_placeholder.empty()
            
            return response
        else:
            # Format the chat history for the model
            formatted_history = self._chat_history[:-1]  # Exclude the just-added message
            
            # Add a system message to ensure proper identity
            system_message = {
                "role": "system", 
                "content": """You are DeepDrone, an advanced AI assistant designed to help with drone operations and data analysis. You are NOT Qwen or any other general AI assistant. Always identify yourself as DeepDrone when asked about your identity. Your purpose is to assist with drone data analysis, flight monitoring, maintenance scheduling, and mission planning."""
            }
            
            # Include the system message and user message
            model_messages = [system_message] + formatted_history
            model_messages.append({"role": "user", "content": message})
            
            # Get response from the model - will be a Message object
            model_response = self.model(model_messages)
            
            # Get the content from the Message object
            response = model_response.content
        
        # Add the response to history
        self.add_to_chat_history("assistant", response)
        
        return response

@tool
def analyze_flight_path(flight_id: str = None) -> str:
    """Analyze a drone's flight path for a specific flight.
    
    Args:
        flight_id: The identifier for the flight to analyze
        
    Returns:
        str: Analysis of the flight path including distance, duration, and altitude changes
    """
    if flight_id is None or flight_id not in tool.agent.flight_logs:
        return "未找到飞行ID。请提供有效的飞行ID。"
    
    flight_data = tool.agent.flight_logs[flight_id]
    
    # Calculate basic flight statistics
    flight_duration = (flight_data['timestamp'].max() - flight_data['timestamp'].min()).total_seconds()
    max_altitude = flight_data['altitude'].max()
    avg_speed = flight_data['speed'].mean() if 'speed' in flight_data.columns else "Not available"
    
    # Generate a path visualization
    plt.figure(figsize=(10, 6))
    
    # Set dark style for the plot
    plt.style.use('dark_background')
    
    if 'latitude' in flight_data.columns and 'longitude' in flight_data.columns:
        plt.plot(flight_data['longitude'], flight_data['latitude'], color='#00ff00')  # Green line
        plt.title(f'Flight Path: {flight_id}', color='white')
        plt.xlabel('Longitude', color='white')
        plt.ylabel('Latitude', color='white')
        plt.tick_params(colors='white')
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black')
        plt.close()
        path_img = base64.b64encode(buf.getvalue()).decode()
    else:
        path_img = None
    
    # Return analysis
    analysis = {
        'flight_id': flight_id,
        'duration_seconds': flight_duration,
        'max_altitude_meters': max_altitude,
        'avg_speed': avg_speed,
        'visualization': path_img
    }
    
    return str(analysis)

@tool
def check_sensor_readings(sensor_name: str = None) -> str:
    """Check the readings from a specific drone sensor.
    
    Args:
        sensor_name: The name of the sensor to check
        
    Returns:
        str: Analysis of the sensor readings including ranges and anomalies
    """
    if sensor_name is None or sensor_name not in tool.agent.sensor_data:
        return f"Sensor not found. Available sensors: {list(tool.agent.sensor_data.keys())}"
    
    sensor_data = tool.agent.sensor_data[sensor_name]
    
    # Basic statistics
    stats = {
        'mean': sensor_data.mean().to_dict(),
        'min': sensor_data.min().to_dict(),
        'max': sensor_data.max().to_dict(),
    }
    
    # Check for anomalies (values more than 3 std devs from mean)
    anomalies = {}
    for column in sensor_data.select_dtypes(include=[np.number]).columns:
        mean = sensor_data[column].mean()
        std = sensor_data[column].std()
        anomaly_points = sensor_data[(sensor_data[column] > mean + 3*std) | 
                                      (sensor_data[column] < mean - 3*std)]
        if not anomaly_points.empty:
            anomalies[column] = len(anomaly_points)
    
    # Return analysis
    analysis = {
        'sensor_name': sensor_name,
        'statistics': stats,
        'anomalies_detected': anomalies,
        'data_points': len(sensor_data)
    }
    
    return str(analysis)

@tool
def recommend_maintenance(flight_hours: float = None) -> str:
    """Recommend maintenance tasks based on flight hours.
    
    Args:
        flight_hours: The number of flight hours since last maintenance
        
    Returns:
        str: Recommended maintenance tasks
    """
    if flight_hours is None:
        return "Please provide the total flight hours for the drone."
    
    recommendations = []
    
    if flight_hours < 10:
        recommendations.append("Regular pre-flight checks only")
    elif 10 <= flight_hours < 50:
        recommendations.append("Basic maintenance check recommended")
        recommendations.append("Inspect propellers and motors")
        recommendations.append("Check battery health")
    elif 50 <= flight_hours < 100:
        recommendations.append("Intermediate maintenance required")
        recommendations.append("Replace propellers")
        recommendations.append("Test all sensors")
        recommendations.append("Firmware updates if available")
    else:
        recommendations.append("Full maintenance overhaul required")
        recommendations.append("Motor inspection and possible replacement")
        recommendations.append("Full electronic systems check")
        recommendations.append("Battery replacement recommended")
        recommendations.append("Structural integrity evaluation")
    
    return "\n".join(recommendations)

@tool
def generate_mission_plan(mission_type: str = None, duration_minutes: float = None) -> str:
    """Generate a mission plan based on the specified type and duration.
    
    Args:
        mission_type: The type of mission (survey, inspection, delivery, etc.)
        duration_minutes: The expected duration of the mission in minutes
        
    Returns:
        str: A mission plan with waypoints and tasks
    """
    if mission_type is None:
        return "请指定任务类型（如：survey, inspection, delivery等）"
    
    if duration_minutes is None:
        return "Please specify the expected mission duration in minutes."
    
    # Generate an appropriate mission plan based on type and duration
    plan = {
        "mission_type": mission_type,
        "duration_minutes": duration_minutes,
        "battery_required": f"{duration_minutes * 1.3:.1f} minutes capacity",
        "pre_flight_checks": [
            "Battery charge level",
            "Motor functionality",
            "GPS signal strength",
            "Camera/sensor calibration"
        ]
    }
    
    # Add mission-specific details
    if mission_type.lower() == "survey":
        plan["flight_pattern"] = "Grid pattern with 70% overlap"
        plan["recommended_altitude"] = "40-60 meters"
        plan["special_considerations"] = "Ensure consistent lighting conditions"
    elif mission_type.lower() == "inspection":
        plan["flight_pattern"] = "Orbital with variable radius"
        plan["recommended_altitude"] = "5-20 meters"
        plan["special_considerations"] = "Maintain safe distance from structures"
    elif mission_type.lower() == "delivery":
        plan["flight_pattern"] = "Direct point-to-point"
        plan["recommended_altitude"] = "30 meters"
        plan["special_considerations"] = "Check payload weight and balance"
    else:
        plan["flight_pattern"] = "Custom"
        plan["recommended_altitude"] = "Dependent on mission specifics"
        plan["special_considerations"] = "Consult regulations for specific operation type"
    
    return str(plan)

# DroneKit real-world control tools

@tool
def connect_to_real_drone(connection_string: str = None) -> str:
    """Connect to a real drone using DroneKit.
    
    Args:
        connection_string: Connection string for the drone (e.g., 'udp:127.0.0.1:14550' for SITL,
                          '/dev/ttyACM0' for serial, or 'tcp:192.168.1.1:5760' for remote connection)
        
    Returns:
        str: Status of the connection
    """
    if connection_string is None:
        return "错误: 需要连接字符串。例如: 'udp:127.0.0.1:14550'（仿真），'/dev/ttyACM0'（串口），或 'tcp:192.168.1.1:5760'（WiFi）"
    
    try:
        # Update mission status
        st.session_state.mission_in_progress = True
        update_mission_status("CONNECTING", f"Connecting to drone at {connection_string}")
        
        success = drone_control.connect_drone(connection_string)
        if success:
            # Get and store current status
            location = drone_control.get_location()
            battery = drone_control.get_battery()
            
            # Update mission status
            update_mission_status("CONNECTED", "Drone connected successfully")
            
            # Format a nice response
            response = {
                "status": "连接成功",
                "location": location,
                "battery": battery
            }
            return str(response)
        else:
            st.session_state.mission_in_progress = False
            update_mission_status("ERROR", "Connection failed")
            return "连接无人机失败。请检查连接字符串并确保无人机已开机。"
    except Exception as e:
        st.session_state.mission_in_progress = False
        update_mission_status("ERROR", f"Connection error: {str(e)}")
        return f"连接无人机出错: {str(e)}"

@tool
def drone_takeoff(altitude: float = None) -> str:
    """Take off to the specified altitude.
    
    Args:
        altitude: Target altitude in meters
        
    Returns:
        str: Status of the takeoff
    """
    if altitude is None:
        return "Error: Altitude is required. Specify a safe takeoff altitude in meters."
    
    try:
        # Check if mission was interrupted
        if st.session_state.interrupt_mission:
            st.session_state.interrupt_mission = False
            return "Takeoff aborted due to mission interrupt request"
        
        # Update mission status
        update_mission_status("TAKING OFF", f"起飞到 {altitude} 米")
        
        success = drone_control.takeoff(altitude)
        if success:
            update_mission_status("AIRBORNE", f"已到达目标高度 {altitude} 米")
            return f"起飞成功！已到达目标高度 {altitude} 米。"
        else:
            update_mission_status("ERROR", "起飞失败")
            return "起飞失败。请确保已连接无人机且处于安全起飞区域。"
    except Exception as e:
        update_mission_status("ERROR", f"起飞出错: {str(e)}")
        return f"起飞过程中出错: {str(e)}"

@tool
def drone_land() -> str:
    """Land the drone.
    
    Returns:
        str: Status of the landing
    """
    try:
        # Update mission status
        update_mission_status("LANDING", "无人机正在降落")
        
        success = drone_control.land()
        if success:
            update_mission_status("LANDED", "无人机已降落")
            st.session_state.mission_in_progress = False
            return "降落指令发送成功。无人机已降落。"
        else:
            update_mission_status("ERROR", "降落失败")
            return "降落指令失败。请确保已连接无人机。"
    except Exception as e:
        update_mission_status("ERROR", f"降落出错: {str(e)}")
        return f"降落过程中出错: {str(e)}"

@tool
def drone_return_home() -> str:
    """Return the drone to its launch location.
    
    Returns:
        str: Status of the return-to-home command
    """
    try:
        # Update mission status
        update_mission_status("RETURNING", "返回起飞点")
        
        success = drone_control.return_home()
        if success:
            update_mission_status("RETURNING", "无人机正在返航")
            return "返航指令发送成功。无人机正在返回起飞点。"
        else:
            update_mission_status("ERROR", "返航失败")
            return "返航指令失败。请确保已连接无人机。"
    except Exception as e:
        update_mission_status("ERROR", f"返航出错: {str(e)}")
        return f"返航过程中出错: {str(e)}"

@tool
def drone_fly_to(latitude: float = None, longitude: float = None, altitude: float = None) -> str:
    """Fly the drone to a specific GPS location.
    
    Args:
        latitude: Target latitude in degrees
        longitude: Target longitude in degrees
        altitude: Target altitude in meters
        
    Returns:
        str: Status of the goto command
    """
    if latitude is None or longitude is None or altitude is None:
        return "错误: 纬度、经度和高度均为必填项。"
    
    try:
        success = drone_control.fly_to(latitude, longitude, altitude)
        if success:
            return f"指令发送成功。飞往: 纬度 {latitude}, 经度 {longitude}, 高度 {altitude}米"
        else:
            return "指令失败。请确保已连接无人机并处于GUIDED模式。"
    except Exception as e:
        return f"飞行指令出错: {str(e)}"

@tool
def get_drone_location() -> str:
    """Get the current GPS location of the drone.
    
    Returns:
        str: Current latitude, longitude, and altitude
    """
    try:
        location = drone_control.get_location()
        return str(location)
    except Exception as e:
        return f"获取无人机位置出错: {str(e)}"

@tool
def get_drone_battery() -> str:
    """Get the current battery level of the drone.
    
    Returns:
        str: Current battery voltage and percentage
    """
    try:
        battery = drone_control.get_battery()
        return str(battery)
    except Exception as e:
        return f"获取电池状态出错: {str(e)}"

@tool
def execute_drone_mission(waypoints: List[Dict[str, float]] = None) -> str:
    """Upload and execute a mission with multiple waypoints.
    
    Args:
        waypoints: List of dictionaries with lat, lon, alt for each waypoint
            Example: [{"lat": 37.123, "lon": -122.456, "alt": 30}, {"lat": 37.124, "lon": -122.457, "alt": 50}]
        
    Returns:
        str: Status of the mission execution
    """
    if waypoints is None or not isinstance(waypoints, list) or len(waypoints) == 0:
        return "错误: 需要航点列表。每个航点需包含lat, lon, alt键。"
    
    # Validate each waypoint
    for i, wp in enumerate(waypoints):
        if not all(key in wp for key in ["lat", "lon", "alt"]):
            return f"错误: 航点 {i} 缺少必要字段。每个航点必须有lat, lon, alt。"
    
    try:
        # Update mission status
        update_mission_status("MISSION", f"开始任务，共 {len(waypoints)} 个航点")
        
        # Check for mission interrupt before starting
        if st.session_state.interrupt_mission:
            st.session_state.interrupt_mission = False
            update_mission_status("ABORTED", "任务在执行前被中断")
            return "任务因中断请求已取消"
        
        # Execute mission with progress updates
        success = drone_control.execute_mission_plan(waypoints)
        
        # Simulate mission progress (in a real implementation, you'd get actual progress from the drone)
        if success:
            total_waypoints = len(waypoints)
            for i in range(total_waypoints):
                # Check for interrupt between waypoints
                if st.session_state.interrupt_mission:
                    st.session_state.interrupt_mission = False
                    update_mission_status("INTERRUPTED", "Mission interrupted, returning to base")
                    drone_control.return_home()
                    time.sleep(2)
                    update_mission_status("RETURNED", "Drone returned to base after interrupt")
                    return f"Mission interrupted after waypoint {i+1}/{total_waypoints}. Drone returned to base."
                
                # Update status for current waypoint
                wp = waypoints[i]
                update_mission_status(
                    "EXECUTING MISSION", 
                    f"飞往航点 {i+1}/{total_waypoints}: 纬度={wp['lat']:.4f}, 经度={wp['lon']:.4f}, 高度={wp['alt']}米"
                )
                
                # Simulate time taken to reach waypoint
                time.sleep(2)
            
            # Mission completed successfully
            update_mission_status("MISSION COMPLETE", "所有航点已到达")
            return f"任务完成，共 {len(waypoints)} 个航点。"
        else:
            update_mission_status("ERROR", "任务执行失败")
            return "任务执行失败。请确保已连接无人机。"
    except Exception as e:
        update_mission_status("ERROR", f"任务出错: {str(e)}")
        return f"任务执行出错: {str(e)}"

@tool
def disconnect_from_drone() -> str:
    """Disconnect from the drone.
    
    Returns:
        str: Status of the disconnection
    """
    try:
        # Update mission status
        update_mission_status("DISCONNECTING", "正在断开无人机连接")
        
        drone_control.disconnect_drone()
        st.session_state.mission_in_progress = False
        update_mission_status("STANDBY", "已断开无人机连接")
        return "已成功断开无人机连接。"
    except Exception as e:
        update_mission_status("ERROR", f"断开连接出错: {str(e)}")
        return f"断开无人机连接出错: {str(e)}"

def create_deepseek_model():
    """Create a DeepSeek model instance"""
    # Check if DEEPSEEK_API_KEY is set in environment variables
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not deepseek_api_key:
        st.error("未找到 DeepSeek API 密钥。请设置 DEEPSEEK_API_KEY 环境变量。")
        # Return a placeholder model that returns a fixed response
        class PlaceholderModel:
            def __call__(self, *args, **kwargs):
                from .deepseek_model import Message
                return Message("Authentication error: No DeepSeek API key provided. Please set an API key to use this feature.")
        return PlaceholderModel()
    
    # Use the token from the environment variable
    return DeepSeekModel(
        max_tokens=2096,
        temperature=0.5,
        model_id='deepseek-reasoner'
    )

def display_message(role, content, avatar_map=None):
    """Display a chat message with custom styling."""
    if avatar_map is None:
        avatar_map = {
            "user": "👤", 
            "assistant": "🚁"
        }
    
    if role == "user":
        # User message styling - right aligned with user avatar
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: #1E1E1E; 
                    border: 1px solid #00ff00;
                    border-radius: 5px; 
                    padding: 8px; 
                    margin-bottom: 8px;
                    text-align: right;
                    max-width: 90%;
                    float: right;
                    color: #FFFFFF;
                ">
                    {content}
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(f"<div style='font-size: 20px; text-align: center; color: #00ff00;'>{avatar_map['user']}</div>", unsafe_allow_html=True)
    else:
        # Assistant message styling - left aligned with drone avatar
        col1, col2 = st.columns([1, 6])
        with col1:
            st.markdown(f"<div style='font-size: 20px; text-align: center; color: #00ff00;'>{avatar_map['assistant']}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #101010; 
                    border: 1px solid #00ff00;
                    border-radius: 5px; 
                    padding: 8px; 
                    margin-bottom: 8px;
                    text-align: left;
                    max-width: 90%;
                    color: #00ff00;
                ">
                    {content}
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    # Add a smaller divider to separate messages
    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

def initialize_chat_container():
    """Initialize the chat container with greeting message."""
    if "chat_container" not in st.session_state:
        chat_container = st.container()
        with chat_container:
            # Initialize with greeting message
            display_message(
                "assistant",
                "INITIALIZING DEEP DRONE SYSTEM... ONLINE. How can I assist with your mission today? You can request flight data analysis, sensor readings, maintenance recommendations, or mission planning."
            )
            
        st.session_state.chat_container = chat_container

def main():
    # Ensure all session state variables are initialized
    if 'mission_status' not in st.session_state:
        st.session_state.mission_status = "STANDBY"
    if 'mission_phase' not in st.session_state:
        st.session_state.mission_phase = ""
    if 'mission_in_progress' not in st.session_state:
        st.session_state.mission_in_progress = False
    if 'interrupt_mission' not in st.session_state:
        st.session_state.interrupt_mission = False
    if 'mission_log' not in st.session_state:
        st.session_state.mission_log = []
    
    # Add custom CSS for proper layout
    st.markdown("""
    <style>
    /* Remove padding from the main container */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }
    
    /* Dark background for the entire app */
    .stApp {
        background-color: #0a1929 !important;
        color: #00ffff !important;
        margin: 0 !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    /* Dark background for main content */
    .main .block-container {
        background-color: #0a1929 !important;
    }
    
    /* Dark styling for sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 25, 41, 0.9) !important;
        border-right: 1px solid #00ffff !important;
        backdrop-filter: blur(10px);
    }
    
    /* Dark styling for all inputs */
    .stTextInput > div {
        background-color: rgba(10, 25, 41, 0.9) !important;
        color: #00ffff !important;
        border: 1px solid #00ffff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2) !important;
    }
    
    .stTextInput input {
        color: #00ffff !important;
        background-color: rgba(10, 25, 41, 0.9) !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    .stTextInput input::placeholder {
        color: rgba(0, 255, 255, 0.5) !important;
    }
    
    /* Override all Streamlit default styling */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #00ffff !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Command bar fixed at bottom */
    .command-bar-wrapper {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background-color: rgba(10, 25, 41, 0.9) !important;
        border-top: 2px solid #00ffff !important;
        padding: 10px !important;
        z-index: 9999 !important;
        width: 100% !important;
        backdrop-filter: blur(10px);
    }
    
    /* Chat container */
    .chat-container {
        height: calc(100vh - 300px) !important;
        max-height: 300px !important;
        overflow-y: auto !important;
        padding: 15px !important;
        margin-bottom: 30px !important;
        background-color: rgba(10, 25, 41, 0.9) !important;
        border: 1px solid #00ffff;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        display: flex !important;
        flex-direction: column !important;
        backdrop-filter: blur(10px);
    }
    
    /* Override button styling */
    button[kind="secondaryFormSubmit"] {
        background-color: rgba(10, 25, 41, 0.9) !important;
        color: #00ffff !important;
        border: 1px solid #00ffff !important;
        border-radius: 5px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: bold !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="secondaryFormSubmit"]:hover {
        background-color: #00ffff !important;
        color: #0a1929 !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4) !important;
    }
    
    .stButton > button {
        background-color: rgba(10, 25, 41, 0.9) !important;
        color: #00ffff !important;
        border: 1px solid #00ffff !important;
        border-radius: 5px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: bold !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #00ffff !important;
        color: #0a1929 !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4) !important;
    }
    
    /* Hide Streamlit's default footer */
    footer, header {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Futuristic text styling */
    .futuristic-text {
        font-family: "Orbitron", sans-serif !important;
        color: #00ffff !important;
        font-weight: bold !important;
        text-shadow: 0 0 10px #00ffff !important;
    }
    
    /* Styling for subheader */
    .subheader {
        color: #00ffff !important;
        font-family: "Orbitron", sans-serif !important;
    }
    
    /* Force dark background for body */
    body {
        background-color: #0a1929 !important;
    }
    
    /* Override any Streamlit white backgrounds */
    .css-1kyxreq, .css-12oz5g7, .css-1r6slb0, .css-1n76uvr, .css-18e3th9 {
        background-color: #0a1929 !important;
    }
    
    /* Fix header text color */
    .css-10trblm {
        color: #00ffff !important;
    }
    
    /* Ensure the header is cyan */
    h1 {
        color: #00ffff !important;
        font-family: "Orbitron", sans-serif !important;
        text-shadow: 0 0 10px #00ffff !important;
    }
    
    /* Add a glow effect to cyan text for a more futuristic feel */
    .glow-text {
        text-shadow: 0 0 10px #00ffff !important;
    }

    /* Override more styles to ensure everything is dark */
    div[data-baseweb="base-input"] {
        background-color: rgba(10, 25, 41, 0.9) !important;
    }

    div[data-baseweb="input"] {
        background-color: rgba(10, 25, 41, 0.9) !important;
    }
    
    /* Fix for dark form backgrounds */
    [data-testid="stForm"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    
    .status-active {
        background-color: #00ffff;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Futuristic header with glow effect
    st.markdown("<h1 class='glow-text' style='text-align: center; color: #00ffff; font-family: \"Orbitron\", sans-serif; margin-top: 0; margin-bottom: 5px;'>DeepDrone 指挥中心</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader glow-text' style='text-align: center; margin-bottom: 5px;'>AI 驱动的高级无人机作业平台</p>", unsafe_allow_html=True)
    
    # Compact status display inline
    status_cols = st.columns(4)
    with status_cols[0]:
        st.markdown("<div class='futuristic-text' style='font-size: 12px;'><span class='status-indicator status-active'></span><b>系统:</b> 在线</div>", unsafe_allow_html=True)
    with status_cols[1]:
        st.markdown("<div class='futuristic-text' style='font-size: 12px;'><span class='status-indicator status-active'></span><b>连接:</b> 安全</div>", unsafe_allow_html=True)
    with status_cols[2]:
        st.markdown("<div class='futuristic-text' style='font-size: 12px;'><span class='status-indicator status-active'></span><b>GPS:</b> 活跃</div>", unsafe_allow_html=True)
    with status_cols[3]:
        st.markdown("<div class='futuristic-text' style='font-size: 12px;'><span class='status-indicator status-active'></span><b>加密:</b> 已启用</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 1px solid #00ffff; margin: 5px 0 10px 0;'>", unsafe_allow_html=True)
    
    # Initialize session state for drone assistant and other needed state
    if 'drone_agent' not in st.session_state:
        model = create_deepseek_model()
        st.session_state['drone_agent'] = DroneAssistant(
            tools=[
                # Data analysis tools
                analyze_flight_path, 
                check_sensor_readings,
                recommend_maintenance, 
                generate_mission_plan,
                
                # Drone control tools
                connect_to_real_drone,
                drone_takeoff,
                drone_land,
                drone_return_home,
                drone_fly_to,
                get_drone_location,
                get_drone_battery,
                execute_drone_mission,
                disconnect_from_drone
            ],
            model=model,
            additional_authorized_imports=["pandas", "numpy", "matplotlib"]
        )
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Generate sample data for demo purposes
    if 'demo_data_loaded' not in st.session_state:
        # Sample flight log
        timestamps = pd.date_range(start='2023-01-01', periods=100, freq='10s')
        flight_log = pd.DataFrame({
            'timestamp': timestamps,
            'altitude': np.random.normal(50, 10, 100),
            'speed': np.random.normal(15, 5, 100),
            'latitude': np.linspace(37.7749, 37.7750, 100) + np.random.normal(0, 0.0001, 100),
            'longitude': np.linspace(-122.4194, -122.4192, 100) + np.random.normal(0, 0.0001, 100)
        })
        st.session_state['drone_agent'].register_flight_log('flight_001', flight_log)
        
        # Sample sensor data
        battery_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=50, freq='1min'),
            'voltage': np.random.normal(11.1, 0.2, 50),
            'current': np.random.normal(5, 1, 50),
            'temperature': np.random.normal(30, 5, 50)
        })
        st.session_state['drone_agent'].register_sensor_data('battery', battery_data)
        
        imu_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='1s'),
            'acc_x': np.random.normal(0, 0.5, 1000),
            'acc_y': np.random.normal(0, 0.5, 1000),
            'acc_z': np.random.normal(9.8, 0.5, 1000),
            'gyro_x': np.random.normal(0, 0.1, 1000),
            'gyro_y': np.random.normal(0, 0.1, 1000),
            'gyro_z': np.random.normal(0, 0.1, 1000)
        })
        st.session_state['drone_agent'].register_sensor_data('imu', imu_data)
        
        st.session_state['demo_data_loaded'] = True
    
    # Add mission status section to sidebar with improved visibility
    st.sidebar.markdown("<h3 style='color: #00ffff; font-family: \"Orbitron\", sans-serif; text-shadow: 0 0 10px #00ffff;'>MISSION CONTROL</h3>", unsafe_allow_html=True)
    
    # Dynamic status display that changes color based on status
    status_color = "#00ffff"  # Default cyan
    if st.session_state.mission_status == "ERROR":
        status_color = "#ff3366"  # Red for errors
    elif st.session_state.mission_status in ["CONNECTING", "TAKING OFF", "LANDING", "RETURNING"]:
        status_color = "#ffcc00"  # Yellow for transitions
    elif st.session_state.mission_status in ["MISSION", "EXECUTING MISSION", "AIRBORNE"]:
        status_color = "#00ffff"  # Cyan for active mission
    
    st.sidebar.markdown(f"""
    <div style='font-family: "Orbitron", sans-serif; color: #00ffff; background-color: rgba(10, 25, 41, 0.9); padding: 15px; border-radius: 10px; border: 1px solid #00ffff; box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);'>
        <div style='margin-bottom: 10px;'><span class='status-indicator status-active'></span><b>状态:</b> <span style="color: {status_color}; font-weight: bold;">{st.session_state.mission_status}</span></div>
        <div style='margin-bottom: 10px;'><span class='status-indicator status-active'></span><b>阶段:</b> <span style="color: {status_color};">{st.session_state.mission_phase}</span></div>
        <div style='margin-bottom: 10px;'><span class='status-indicator status-active'></span><b>任务中:</b> {"是" if st.session_state.mission_in_progress else "否"}</div>
        <div><span class='status-indicator status-active'></span><b>信号:</b> 强</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add interrupt button if a mission is in progress
    if st.session_state.mission_in_progress:
        if st.sidebar.button("⚠️ 中止任务", 
                            key="abort_button", 
                            help="立即中止当前任务并让无人机返航",
                            type="primary"):
            interrupt_mission()
    
    # Add mission summary in sidebar
    st.sidebar.markdown("<div style='color: #00ffff; font-family: \"Orbitron\", sans-serif; font-size: 12px; margin-top: 20px;'><b>任务消息:</b> 显示在聊天窗口</div>", unsafe_allow_html=True)
    
    # Show just the last message if there are any mission logs
    if st.session_state.mission_log:
        last_entry = st.session_state.mission_log[-1]
        entry_style = ""
        
        # Style the last entry based on its content
        if "ERROR" in last_entry:
            entry_style = "color: #ff3366;"
        elif any(status in last_entry for status in ["CONNECTING", "TAKING OFF", "LANDING", "RETURNING"]):
            entry_style = "color: #ffcc00;"
        elif any(status in last_entry for status in ["MISSION", "EXECUTING", "AIRBORNE"]):
            entry_style = "color: #00ffff;"
        else:
            entry_style = "color: #00ffff;"
            
        st.sidebar.markdown(f"""<div style='font-family: \"Orbitron\", sans-serif; font-size: 11px; {entry_style} background-color: rgba(10, 25, 41, 0.9); padding: 8px; border-radius: 5px; border: 1px solid #00ffff; box-shadow: 0 0 10px rgba(0, 255, 255, 0.1);'><span class='status-indicator status-active'></span>最新: {last_entry}</div>""", unsafe_allow_html=True)
    
    st.sidebar.markdown("<hr style='border: 1px solid #00ffff; margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Command reference
    st.sidebar.markdown("<h3 style='color: #00ffff; font-family: \"Orbitron\", sans-serif; text-shadow: 0 0 10px #00ffff;'>命令参考</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='font-family: "Orbitron", sans-serif; color: #00ffff; background-color: rgba(10, 25, 41, 0.9); padding: 15px; border-radius: 10px; border: 1px solid #00ffff; box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);'>
        <div style='margin-bottom: 10px;'><b>数据分析:</b><br>- "分析 flight_001"<br>- "检查电池传感器读数"<br>- "推荐 75 小时飞行后的维护"</div>
        <div style='margin-bottom: 10px;'><b>任务规划:</b><br>- "创建正方形模式的飞行计划"<br>- "规划 30 分钟的测绘任务"<br>- "连接仿真器，起飞，执行简单正方形飞行并返航"</div>
        <div><b>正确的函数名:</b><br>- connect_to_real_drone()<br>- drone_takeoff()<br>- drone_land()<br>- drone_return_home()<br>- drone_fly_to()<br>- execute_drone_mission()</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<hr style='border: 1px solid #00ffff; margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Available data
    st.sidebar.markdown("<h3 style='color: #00ffff; font-family: \"Orbitron\", sans-serif; text-shadow: 0 0 10px #00ffff;'>可用数据</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='font-family: "Orbitron", sans-serif; color: #00ffff; background-color: rgba(10, 25, 41, 0.9); padding: 15px; border-radius: 10px; border: 1px solid #00ffff; box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);'>
        <div style='margin-bottom: 10px;'><span class='status-indicator status-active'></span><b>飞行日志:</b> flight_001</div>
        <div><span class='status-indicator status-active'></span><b>传感器:</b> battery, imu</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create chat area with container class
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Info message about mission logs appearing in chat
    if st.session_state.mission_in_progress:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 8px; font-family: 'Orbitron', sans-serif; font-size: 12px; color: #00ffff; text-shadow: 0 0 10px #00ffff;"><span class='status-indicator status-active'></span>任务日志将显示在此聊天窗口</div>
        """, unsafe_allow_html=True)
    
    # Display initial assistant greeting or chat history
    if not st.session_state['chat_history']:
        # Welcome message with drone emoji
        st.markdown("""
        <div style="display: flex; align-items: flex-start; margin-bottom: 8px;">
            <div style="font-size: 24px; margin-right: 8px; color: #00ffff; text-shadow: 0 0 10px #00ffff;">🚁</div>
            <div style="background-color: rgba(10, 25, 41, 0.9); border: 1px solid #00ffff; border-radius: 10px; padding: 12px; color: #00ffff; flex-grow: 1; font-family: 'Orbitron', sans-serif; box-shadow: 0 0 15px rgba(0, 255, 255, 0.1); backdrop-filter: blur(5px);">
                DeepDrone 已上线。我是您的AI无人机作业助手。请问有什么可以帮您？您可以请求飞行数据分析、传感器读取、维护建议或任务规划。
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display all messages in history
        for message in st.session_state['chat_history']:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start; justify-content: flex-end; margin-bottom: 8px;">
                    <div style="background-color: rgba(10, 25, 41, 0.9); border: 1px solid #00ffff; border-radius: 10px; padding: 12px; color: #FFFFFF; max-width: 85%; font-family: 'Orbitron', sans-serif; box-shadow: 0 0 15px rgba(0, 255, 255, 0.1); backdrop-filter: blur(5px);">
                        {message["content"]}
                    </div>
                    <div style="font-size: 24px; margin-left: 8px; color: #00ffff; text-shadow: 0 0 10px #00ffff;">👤</div>
                </div>
                """, unsafe_allow_html=True)
            elif message["role"] == "system":
                # System messages (logs) have a different style - centered and distinctive
                st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; margin: 4px 0;">
                    <div style="background-color: rgba(10, 25, 41, 0.9); border: 1px solid #00ffff; border-radius: 5px; padding: 8px 15px; font-family: 'Orbitron', sans-serif; font-size: 12px; text-align: center; max-width: 90%; box-shadow: 0 0 15px rgba(0, 255, 255, 0.1); backdrop-filter: blur(5px);"><span class='status-indicator status-active'></span>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start; margin-bottom: 8px;">
                    <div style="font-size: 24px; margin-right: 8px; color: #00ffff; text-shadow: 0 0 10px #00ffff;">🚁</div>
                    <div style="background-color: rgba(10, 25, 41, 0.9); border: 1px solid #00ffff; border-radius: 10px; padding: 12px; color: #00ffff; max-width: 85%; font-family: 'Orbitron', sans-serif; box-shadow: 0 0 15px rgba(0, 255, 255, 0.1); backdrop-filter: blur(5px);">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Display the last image if there is one
    if 'last_image' in st.session_state:
        st.image(f"data:image/png;base64,{st.session_state['last_image']}")
        # Clear the image from session state after displaying
        del st.session_state['last_image']
        
    # Close the chat-container div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Minimal spacing for the command bar
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Command bar fixed at the bottom
    st.markdown("""
    <div class="command-bar-wrapper" style="position: fixed; bottom: 0; left: 0; right: 0; padding: 12px; background-color: rgba(10, 25, 41, 0.9); border-top: 1px solid #00ffff; box-shadow: 0 -5px 20px rgba(0, 255, 255, 0.1); backdrop-filter: blur(10px);">
    """, unsafe_allow_html=True)
    
    # Create a more compact form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_message = st.text_input(
                "COMMAND:",
                placeholder="请输入您的指令...",
                label_visibility="collapsed",
                key="command_input"
            )
        with col2:
            submit_button = st.form_submit_button(
                "执行", 
                use_container_width=True
            )
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle user input from chat
    if submit_button and user_message:
        # Add user message to chat history
        st.session_state['chat_history'].append({
            'role': 'user',
            'content': user_message
        })
        
        # Process with the agent
        with st.spinner('处理中...'):
            # Check for identity questions directly
            identity_patterns = ["who are you", "what are you", "your name", "introduce yourself"]
            if any(pattern in user_message.lower() for pattern in identity_patterns):
                response = "我是 DeepDrone，一个专为无人机操作与数据分析设计的AI助手。我可以协助飞行数据分析、传感器读取、维护建议和任务规划。"
            else:
                # Process through the agent's chat method
                response = st.session_state['drone_agent'].chat(user_message)
                # No need to handle Message objects here as that's handled inside the chat method
            
            # Handle base64 images in responses
            if isinstance(response, str) and "visualization" in response and "base64" in response:
                # Extract and display the image
                import re
                import ast
                
                try:
                    # Parse the response to extract the base64 image
                    response_dict = ast.literal_eval(response)
                    if isinstance(response_dict, dict) and 'visualization' in response_dict:
                        img_data = response_dict['visualization']
                        if img_data:
                            # Store the image in session state to display on next rerun
                            st.session_state['last_image'] = img_data
                            # Remove the image data from the text response
                            response_dict['visualization'] = "[FLIGHT PATH VISUALIZATION DISPLAYED]"
                            response = str(response_dict)
                except (SyntaxError, ValueError):
                    # If parsing fails, just display the text
                    pass
            
            # Add assistant response to chat history
            st.session_state['chat_history'].append({
                'role': 'assistant',
                'content': response
            })
        
        # Rerun to refresh the page and display the new messages
        st.rerun()

if __name__ == "__main__":
    main() 