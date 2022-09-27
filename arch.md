# MLHub - 架构

## 后端 Flask

Python-Flask作为后端，提供基于web请求的业务服务。

上传图片

```http
POST /api/upload/
```

使用`<model>`模型预测图片`<image>`

```http
POST /ml/dispatch/<model>/<image>
```

## 前端 Vue

基于Vue的web前端