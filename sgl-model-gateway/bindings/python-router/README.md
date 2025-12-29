# sglang-router (DEPRECATED)

**This package has been renamed to `sgl-model-gateway`.**

## Migration

Please update your dependencies to use `sgl-model-gateway` instead:

```bash
pip uninstall sglang-router
pip install sgl-model-gateway
```

And update your imports:

```python
# Old (deprecated)
from sglang_router import Router, RouterArgs

# New
from sgl_model_gateway import Router, RouterArgs
```

This compatibility package will continue to work but emits deprecation warnings. It will be removed in a future release.

## About sgl-model-gateway

sgl-model-gateway is a high-performance Rust-based load balancer for SGLang with multiple routing algorithms and prefill-decode disaggregation support.

For documentation, please visit: https://github.com/sgl-project/sglang
