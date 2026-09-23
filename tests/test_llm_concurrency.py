from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import sleep


def test_shared_llm_gateway_never_runs_more_than_five_calls_at_once():
    from app.llm_gateway import invoke_llm

    active = 0
    peak = 0
    lock = Lock()
    five_started = Event()

    class SlowLLM:
        def invoke(self, value):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 5:
                    five_started.set()
            five_started.wait(timeout=1)
            sleep(0.03)
            with lock:
                active -= 1
            return value

    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(lambda n: invoke_llm(SlowLLM(), n), range(15)))

    assert results == list(range(15))
    assert peak == 5
