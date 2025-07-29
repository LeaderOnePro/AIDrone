#!/usr/bin/env python3
"""
测试GLM模型的简单脚本
"""
import os
from drone.glm_model import GLMModel, Message

def test_glm_model():
    """测试GLM模型基本功能"""
    print("开始测试GLM模型...")
    
    # 检查API密钥
    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到GLM_API_KEY环境变量")
        print("请设置GLM_API_KEY环境变量后重试")
        return False
    
    try:
        # 创建模型实例
        model = GLMModel(
            model_id='glm-4-plus',
            max_tokens=100,
            temperature=0.5
        )
        print("✅ GLM模型实例创建成功")
        
        # 测试简单文本生成
        test_prompt = "你好，请简单介绍一下你自己。"
        print(f"测试提示: {test_prompt}")
        
        response = model(test_prompt)
        
        if isinstance(response, Message):
            print("✅ 模型响应成功")
            print(f"响应内容: {response.content}")
            return True
        else:
            print("❌ 模型响应格式错误")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_glm_model()
    if success:
        print("\n🎉 GLM模型测试通过！")
    else:
        print("\n💥 GLM模型测试失败！")