from visual_service.errors import (
    InvalidRequest,
    ModelNotReady,
    ServiceError,
    WindowMismatch,
    _error_response,
    _is_oom,
)


class TestErrorResponse:
    def test_structure(self):
        resp = _error_response(400, "INVALID_REQUEST", "bad input", retryable=False)
        assert resp.status_code == 400
        body = resp.body
        import json
        data = json.loads(body)
        assert data == {
            "error": {
                "code": "INVALID_REQUEST",
                "message": "bad input",
                "retryable": False,
            }
        }

    def test_retryable_true(self):
        import json
        resp = _error_response(503, "MODEL_NOT_READY", "not loaded", retryable=True)
        data = json.loads(resp.body)
        assert data["error"]["retryable"] is True


class TestIsOom:
    def test_regular_runtime_error(self):
        assert _is_oom(RuntimeError("something broke")) is False

    def test_runtime_error_with_oom_message(self):
        assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) is True

    def test_exception_named_out_of_memory_error(self):
        class OutOfMemoryError(RuntimeError):
            pass
        assert _is_oom(OutOfMemoryError("GPU OOM")) is True

    def test_chained_oom(self):
        oom = RuntimeError("CUDA out of memory")
        wrapper = RuntimeError("Inference failed")
        wrapper.__cause__ = oom
        assert _is_oom(wrapper) is True

    def test_chained_non_oom(self):
        inner = ValueError("bad value")
        wrapper = RuntimeError("wrapped")
        wrapper.__cause__ = inner
        assert _is_oom(wrapper) is False

    def test_no_cause(self):
        assert _is_oom(ValueError("no cause")) is False


class TestExceptionClassAttributes:
    def test_model_not_ready(self):
        exc = ModelNotReady("not loaded")
        assert exc.status_code == 503
        assert exc.error_code == "MODEL_NOT_READY"
        assert exc.retryable is True

    def test_invalid_request(self):
        exc = InvalidRequest("bad profile")
        assert exc.status_code == 400
        assert exc.error_code == "INVALID_REQUEST"
        assert exc.retryable is False

    def test_window_mismatch(self):
        exc = WindowMismatch("start >= end")
        assert exc.status_code == 409
        assert exc.error_code == "WINDOW_MISMATCH"
        assert exc.retryable is False

    def test_base_service_error(self):
        exc = ServiceError("generic")
        assert exc.status_code == 500
        assert exc.error_code == "INFERENCE_FAILED"
        assert exc.retryable is True
