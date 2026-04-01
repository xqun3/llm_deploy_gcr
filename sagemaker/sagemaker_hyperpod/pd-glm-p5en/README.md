# GLM-5-FP8 PD 分离部署指南 (p5en.48xlarge)

## 概述

在 SageMaker HyperPod EKS 上使用 SGLang PD (Prefill-Decode) 分离架构部署 GLM-5-FP8 (754B MoE, 40B 激活参数)。

## 架构

```
客户端 -> Router (端口 8000, API Key 认证)
              |
       +------+------+
       |              |
  Prefill 节点     Decode 节点
  (p5en #1, TP=8)  (p5en #2, TP=8)
  端口 30000        端口 30001
  8x H200 141GB     8x H200 141GB
       |              |
       +-- NIXL/EFA --+
       (KV Cache 传输)
```

- **Prefill 节点**: 处理输入 token, 计算密集型
- **Decode 节点**: 逐 token 生成, 显存带宽密集型
- **Router**: 请求路由, 分发 prefill/decode 任务
- **KV Cache 传输**: 通过 NIXL 在节点间传输

## 环境

| 组件 | 详情 |
|------|------|
| 集群 | `hp-cluster-hypd-ohio-0225` (HyperPod EKS, us-east-2) |
| EKS | `eks-cluster-hypd-ohio-0225` (K8s 1.33) |
| 实例 | 2x ml.p5en.48xlarge (8x H200 141GB, 16x EFA) |
| 模型 | [zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8) (754B MoE, FP8, ~710GB) |
| 镜像 | `596899493901.dkr.ecr.us-east-2.amazonaws.com/sgl-dev-cu13:20260402013115` |
| 基础镜像 | `lmsysorg/sglang:dev-cu13` + EFA 1.47.0 |
| 存储 | NVMe 本地盘 (`/opt/dlami/nvme/GLM-5-FP8`) |

## 文件说明

| 文件 | 说明 |
|------|------|
| `sgl-pd-glm-flash-p5en.yaml` | K8s 部署清单 (Services, StatefulSets, Router, RBAC) |
| `dockerfile` | 自定义镜像: sglang dev-cu13 + EFA |
| `build-image.sh` | 构建并推送镜像到 ECR |

## 部署步骤

### 1. 构建镜像 (一次性)

```bash
cd pd-glm-p5en
bash build-image.sh
# 输出: 596899493901.dkr.ecr.us-east-2.amazonaws.com/sgl-dev-cu13:<时间戳>
```

### 2. 下载模型到 NVMe (每台 p5en 节点各执行一次)

```bash
# 创建下载 Pod, nodeName 改为目标节点名
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: model-downloader
  namespace: kubeflow
spec:
  restartPolicy: Never
  nodeName: <p5en节点名>
  tolerations:
    - operator: Exists
  containers:
    - name: downloader
      image: 596899493901.dkr.ecr.us-east-2.amazonaws.com/sgl-dev-cu13:<时间戳>
      command: ["/bin/bash", "-c"]
      args:
        - |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          export PATH="/root/.local/bin:\$PATH"
          uvx --from 'huggingface_hub[hf_transfer]' huggingface-cli download zai-org/GLM-5-FP8 --local-dir /nvme/GLM-5-FP8
          echo "下载完成"
          sleep infinity
      volumeMounts:
        - name: nvme
          mountPath: /nvme
  volumes:
    - name: nvme
      hostPath:
        path: /opt/dlami/nvme
        type: DirectoryOrCreate
EOF
```

p5en 节点使用 `hf_transfer` 下载约 4 分钟 (~710GB, 400Gbps 网络)。

### 3. 部署服务

```bash
kubectl apply -f sgl-pd-glm-flash-p5en.yaml
```

### 4. 监控启动

首次启动包含 DeepGEMM JIT 编译 (~10-15 分钟), 后续重启可通过缓存持久化跳过。

```bash
# 监控 Pod 状态
kubectl get pods -n kubeflow -l 'app in (pd-prefill, pd-decode, pd-router)' -w

# 健康检查
kubectl exec pd-prefill-0 -n kubeflow -c pd-prefill -- curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health
kubectl exec pd-decode-0 -n kubeflow -c pd-decode -- curl -s -o /dev/null -w "%{http_code}" http://localhost:30001/health

# 查看日志 (等待 "The server is fired up and ready to roll!")
kubectl logs -f pd-prefill-0 -n kubeflow -c pd-prefill
kubectl logs -f pd-decode-0 -n kubeflow -c pd-decode
```

### 5. 测试调用

```bash
# 端口转发
kubectl port-forward svc/pd-router 8090:8000 -n kubeflow &

# API 调用
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-glm5test" \
  -d '{"model":"/nvme/GLM-5-FP8","messages":[{"role":"user","content":"你好"}],"max_tokens":200}'
```

### 6. Benchmark 测试

```bash
# E2E 测试 (通过 Router)
kubectl exec pd-prefill-0 -n kubeflow -c pd-prefill -- python3 -m sglang.bench_serving \
  --backend sglang --host pd-router --port 8000 \
  --model /nvme/GLM-5-FP8 --dataset-name random \
  --random-input-len 2048 --random-output-len 256 \
  --num-prompts 50 --request-rate 4 \
  --header "x-api-key: sk-glm5test"

# Prefill 单独测试
kubectl exec pd-prefill-0 -n kubeflow -c pd-prefill -- python3 -m sglang.bench_serving \
  --backend sglang --model /nvme/GLM-5-FP8 --dataset-name random \
  --random-input-len 2048 --random-output-len 256 \
  --num-prompts 20 --request-rate 4 \
  --pd-separated --profile-prefill-url http://pd-prefill:30000

# Decode 单独测试
kubectl exec pd-prefill-0 -n kubeflow -c pd-prefill -- python3 -m sglang.bench_serving \
  --backend sglang --model /nvme/GLM-5-FP8 --dataset-name random \
  --random-input-len 2048 --random-output-len 256 \
  --num-prompts 20 --request-rate 4 \
  --pd-separated --profile-decode-url http://pd-decode:30001
```

## Benchmark 结果

### 测试方法

本机拉取 `lmsysorg/sglang:dev-cu13` 镜像, 通过 `kubectl port-forward` 转发 prefill (30000) 和 decode (30001) 端口到本机, 使用 docker 运行 benchmark:

```bash
# 端口转发
kubectl port-forward pod/pd-prefill-0 30000:30000 -n kubeflow &
kubectl port-forward pod/pd-decode-0 30001:30001 -n kubeflow &

# 运行 benchmark
docker run --rm --network host lmsysorg/sglang:dev-cu13 \
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --model /nvme/GLM-5-FP8 --tokenizer zai-org/GLM-5-FP8 \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 256 \
    --num-prompts 50 --request-rate 4
```

## 清理

```bash
kubectl delete -f sgl-pd-glm-flash-p5en.yaml
```
