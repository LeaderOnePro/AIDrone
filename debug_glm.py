#!/usr/bin/env python3
"""
调试GLM模型API响应的脚本
"""
import os
import json
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def debug_glm_api():
    """直接测试GLM API"""
    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到GLM_API_KEY环境变量")
        return
    
    # GLM API配置
    base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "glm-4.5",
        "messages": [
            {"role": "user", "content": "你好，请简单介绍一下你自己。"}
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    
    print("🔄 发送API请求...")
    print(f"URL: {base_url}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(base_url, headers=headers, json=payload)
        print(f"✅ HTTP状态码: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ API响应成功")
            print(f"完整响应: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message = response_data['choices'][0]['message']
                content = message.get('content', '')
                print(f"📝 提取的内容: '{content}'")
            else:
                print("⚠️ 响应中没有choices或choices为空")
        else:
            print(f"❌ API请求失败")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")

if __name__ == "__main__":
    debug_glm_api()
