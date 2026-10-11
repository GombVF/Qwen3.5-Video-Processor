import json
import logging

from visual_service.logging import JSONFormatter, configure_logging, correlation


def _make_record(message="test_event", level=logging.INFO, name="test.logger", extra=None):
    logger = logging.getLogger(name)
    record = logger.makeRecord(name, level, "test.py", 1, message, (), None)
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


class TestJSONFormatter:
    def setup_method(self):
        self.fmt = JSONFormatter()

    def test_outputs_valid_json(self):
        record = _make_record()
        line = self.fmt.format(record)
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_contains_required_fields(self):
        record = _make_record(message="hello", name="my.module")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["message"] == "hello"
        assert parsed["logger"] == "my.module"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_timestamp_is_iso_utc(self):
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        ts = parsed["timestamp"]
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_merges_data_extra(self):
        record = _make_record(extra={"data": {"startMs": 0, "endMs": 30000}})
        parsed = json.loads(self.fmt.format(record))
        assert parsed["startMs"] == 0
        assert parsed["endMs"] == 30000

    def test_ignores_non_dict_data(self):
        record = _make_record(extra={"data": "not a dict"})
        parsed = json.loads(self.fmt.format(record))
        assert "data" not in parsed or parsed.get("data") != "not a dict"

    def test_includes_exception(self):
        try:
            raise ValueError("boom")
        except ValueError:
            record = _make_record()
            record.exc_info = logging.sys.exc_info()

        parsed = json.loads(self.fmt.format(record))
        assert "exception" in parsed
        assert "ValueError: boom" in parsed["exception"]

    def test_warning_level(self):
        record = _make_record(level=logging.WARNING)
        parsed = json.loads(self.fmt.format(record))
        assert parsed["level"] == "WARNING"


class TestCorrelationContext:
    def teardown_method(self):
        correlation.set({})

    def test_correlation_included_in_output(self):
        correlation.set({"runId": "run-42", "windowId": "win-7"})
        fmt = JSONFormatter()
        record = _make_record(message="inside_request")
        parsed = json.loads(fmt.format(record))
        assert parsed["runId"] == "run-42"
        assert parsed["windowId"] == "win-7"

    def test_empty_correlation_adds_nothing(self):
        correlation.set({})
        fmt = JSONFormatter()
        record = _make_record(message="no_context")
        parsed = json.loads(fmt.format(record))
        assert "runId" not in parsed
        assert "windowId" not in parsed

    def test_correlation_and_data_both_present(self):
        correlation.set({"runId": "run-1"})
        fmt = JSONFormatter()
        record = _make_record(extra={"data": {"totalMs": 500}})
        parsed = json.loads(fmt.format(record))
        assert parsed["runId"] == "run-1"
        assert parsed["totalMs"] == 500


class TestConfigureLogging:
    def test_root_logger_has_json_handler(self):
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_sets_level(self):
        configure_logging(level=logging.DEBUG)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        configure_logging(level=logging.INFO)

    def test_noisy_loggers_quieted(self):
        configure_logging()
        assert logging.getLogger("uvicorn.access").level >= logging.WARNING
