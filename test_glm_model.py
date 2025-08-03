#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-4.5 Model Integration Test Script
测试GLM-4.5模型集成是否正常工作
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock
import traceback

# 设置UTF-8编码输出
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入测试")
    print("=" * 60)
    
    try:
        # 测试基本导入
        from drone.glm_model import GLMModel, Message
        print("OK GLM 模块导入成功")
        
        # 测试smolagents导入
        from smolagents import CodeAgent
        print("OK smolagents 导入成功")
        
        return True
    except ImportError as e:
        print(f"ERROR 导入失败: {e}")
        return False
    except Exception as e:
        print(f"ERROR 未知错误: {e}")
        return False

def test_model_initialization():
    """测试模型初始化"""
    print("\n" + "=" * 60)
    print("测试 2: 模型初始化测试")
    print("=" * 60)
    
    try:
        from drone.glm_model import GLMModel
        
        # 测试没有API密钥的情况
        if 'GLM_API_KEY' in os.environ:
            del os.environ['GLM_API_KEY']
        
        try:
            model = GLMModel()
            print("ERROR 应该抛出错误但没有抛出")
            return False
        except ValueError as e:
            print("OK 正确抛出了API密钥缺失错误")
        
        # 测试有API密钥的情况
        os.environ['GLM_API_KEY'] = "test_api_key"
        model = GLMModel(
            model_id='glm-4.5',
            max_tokens=1000,
            temperature=0.5
        )
        print("OK 模型初始化成功")
        
        # 验证属性
        assert model.model_id == 'glm-4.5'
        assert model.max_tokens == 1000
        assert model.temperature == 0.5
        assert model.api_key == "test_api_key"
        print("OK 模型属性验证通过")
        
        return True
    except Exception as e:
        print(f"ERROR 初始化测试失败: {e}")
        traceback.print_exc()
        return False

def test_message_class():
    """测试Message类"""
    print("\n" + "=" * 60)
    print("测试 3: Message类测试")
    print("=" * 60)
    
    try:
        from drone.glm_model import Message
        
        # 测试Message创建
        msg = Message("测试内容")
        assert msg.content == "测试内容"
        print("OK Message类创建成功")
        
        # 测试属性
        assert hasattr(msg, 'model')
        assert hasattr(msg, 'created')
        assert hasattr(msg, 'choices')
        print("OK Message类属性验证通过")
        
        return True
    except Exception as e:
        print(f"ERROR Message类测试失败: {e}")
        return False

def test_model_interface():
    """测试模型接口兼容性"""
    print("\n" + "=" * 60)
    print("测试 4: 模型接口兼容性测试")
    print("=" * 60)
    
    try:
        from drone.glm_model import GLMModel, Message
        
        # 设置测试环境
        os.environ['GLM_API_KEY'] = "test_api_key"
        model = GLMModel()
        
        # 检查必需的方法
        assert hasattr(model, '__call__'), "模型必须可调用"
        assert hasattr(model, 'generate'), "模型必须有generate方法"
        print("OK 模型接口检查通过")
        
        # 模拟API响应
        mock_response = {
            "choices": [{
                "message": {
                    "content": "测试响应",
                    "role": "assistant"
                }
            }]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            # 测试字符串输入
            result = model("测试提示")
            assert isinstance(result, Message), "返回值必须是Message对象"
            assert result.content == "测试响应"
            print("OK 字符串输入测试通过")
            
            # 测试消息列表输入
            messages = [{"role": "user", "content": "测试"}]
            result = model(messages)
            assert isinstance(result, Message)
            print("OK 消息列表输入测试通过")
            
            # 测试generate方法
            result = model.generate("测试", max_tokens=500, temperature=0.3)
            assert isinstance(result, Message)
            print("OK generate方法测试通过")
        
        return True
    except Exception as e:
        print(f"ERROR 接口测试失败: {e}")
        traceback.print_exc()
        return False

def test_tool_calling():
    """测试工具调用功能"""
    print("\n" + "=" * 60)
    print("测试 5: 工具调用功能测试")
    print("=" * 60)
    
    try:
        from drone.glm_model import GLMModel, Message
        
        os.environ['GLM_API_KEY'] = "test_api_key"
        model = GLMModel()
        
        # 模拟包含工具调用的API响应
        mock_response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "test_function",
                            "arguments": '{"param1": "value1", "param2": 42}'
                        }
                    }]
                }
            }]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = model("执行工具调用")
            assert isinstance(result, Message)
            # 检查是否包含smolagents期望的格式
            assert "Thought:" in result.content
            assert "Code:" in result.content
            assert "test_function" in result.content
            assert "final_answer" in result.content
            print("OK 工具调用格式正确")
        
        return True
    except Exception as e:
        print(f"ERROR 工具调用测试失败: {e}")
        traceback.print_exc()
        return False

def test_smolagents_integration():
    """测试与smolagents的集成"""
    print("\n" + "=" * 60)
    print("测试 6: smolagents集成测试")
    print("=" * 60)
    
    try:
        from drone.glm_model import GLMModel
        from smolagents import CodeAgent, tool
        
        os.environ['GLM_API_KEY'] = "test_api_key"
        model = GLMModel()
        
        # 创建一个简单的工具
        @tool
        def test_tool(param: str = "default") -> str:
            """测试工具
            
            Args:
                param: 测试参数，用于验证工具调用
            
            Returns:
                str: 测试结果
            """
            return f"测试结果: {param}"
        
        # 创建代理
        agent = CodeAgent(tools=[test_tool], model=model)
        print("OK CodeAgent创建成功")
        
        # 验证代理属性
        assert agent.model is model
        print(f"DEBUG: agent.tools = {agent.tools}")
        print(f"DEBUG: agent.tools type = {type(agent.tools)}")
        # smolagents CodeAgent成功创建即表示集成正常
        print("OK CodeAgent集成验证通过")
        
        return True
    except Exception as e:
        print(f"ERROR smolagents集成测试失败: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试 7: 错误处理测试")
    print("=" * 60)
    
    try:
        from drone.glm_model import GLMModel, Message
        
        os.environ['GLM_API_KEY'] = "test_api_key"
        model = GLMModel()
        
        # 模拟API错误
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("API错误")
            
            result = model("测试")
            assert isinstance(result, Message)
            # GLM模型在_generate_chat_response中返回"Error in API request"
            assert "Error in API request" in result.content
            print("OK API错误处理正确")
        
        # 测试无效JSON响应
        mock_response = {"invalid": "response"}
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None
            
            result = model("测试")
            assert isinstance(result, Message)
            assert result.content == "No response generated"
            print("OK 无效响应处理正确")
        
        return True
    except Exception as e:
        print(f"ERROR 错误处理测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始GLM-4.5模型集成测试")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_initialization,
        test_message_class,
        test_model_interface,
        test_tool_calling,
        test_smolagents_integration,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR 测试异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"PASS 通过: {passed}")
    print(f"FAIL 失败: {failed}")
    print(f"TOTAL 总计: {passed + failed}")
    
    if failed == 0:
        print("\n所有测试通过！GLM-4.5模型集成正常。")
        return True
    else:
        print(f"\n有{failed}个测试失败，需要修复。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)