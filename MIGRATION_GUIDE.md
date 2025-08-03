# DeepSeek到GLM-4.5迁移指南

本文档描述了如何从DeepSeek模型迁移到GLM-4.5模型。

## 主要变更

### 1. 模型文件
- **新增**: `drone/glm_model.py` - GLM-4.5模型实现
- **保留**: `drone/deepseek_model.py` - 原DeepSeek模型实现（作为参考）

### 2. 环境变量
- **旧**: `DEEPSEEK_API_KEY` 
- **新**: `GLM_API_KEY`

### 3. API端点
- **旧**: `https://api.deepseek.com/v1`
- **新**: `https://open.bigmodel.cn/api/paas/v4/chat/completions`

### 4. 模型ID
- **旧**: `deepseek-reasoner`
- **新**: `glm-4.5`

## 配置更新

### 环境变量设置
更新你的 `.env` 文件：

```bash
# 旧配置（删除或注释）
# DEEPSEEK_API_KEY=your_deepseek_api_key

# 新配置
GLM_API_KEY=your_glm_api_key
```

### API密钥获取
1. 访问 [GLM开放平台](https://open.bigmodel.cn/usercenter/apikeys)
2. 注册账号并创建API密钥
3. 将密钥添加到环境变量中

## 代码变更

### 主要文件修改
1. `drone/drone_chat.py`:
   - 导入从 `deepseek_model` 改为 `glm_model`
   - 函数名从 `create_deepseek_model()` 改为 `create_glm_model()`
   - 环境变量检查从 `DEEPSEEK_API_KEY` 改为 `GLM_API_KEY`

2. `main.py`:
   - 认证界面文本更新
   - 环境变量检查更新

3. `README.md`:
   - 项目描述更新
   - 安装说明更新
   - 技术栈说明更新

4. `.env-example`:
   - 示例环境变量更新

## 测试

运行测试脚本验证迁移是否成功：

```bash
python test_glm_model.py
```

## 兼容性说明

- GLM-4.5模型与原DeepSeek模型在接口上保持兼容
- 所有原有功能应该正常工作
- 响应格式和质量可能略有差异

## 回滚方案

如果需要回滚到DeepSeek模型：

1. 切换回主分支：
   ```bash
   git checkout main
   ```

2. 或者手动修改代码：
   - 将 `glm_model` 导入改回 `deepseek_model`
   - 将 `GLM_API_KEY` 改回 `DEEPSEEK_API_KEY`
   - 将函数调用改回原来的名称

## 注意事项

1. **API配额**: GLM-4.5和DeepSeek有不同的API配额和计费方式
2. **响应时间**: 两个模型的响应时间可能有差异
3. **功能差异**: 某些高级功能可能在不同模型间有差异

## 支持

如果在迁移过程中遇到问题，请：
1. 检查API密钥是否正确设置
2. 确认网络连接正常
3. 查看错误日志获取详细信息