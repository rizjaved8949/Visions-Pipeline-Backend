import threading
import uuid

from concurrent.futures import (
    ThreadPoolExecutor,
)

from .config import (
    MAX_WORKERS,
)

from .pipeline import (
    KitchenPipeline,
)

from .storage import (
    STORE,
)


class KitchenService:

    def __init__(self):

        self.executor = (
            ThreadPoolExecutor(
                max_workers=
                    MAX_WORKERS
            )
        )

        self.stop_events = {}

        self.lock = (
            threading.Lock()
        )


    def _new_session_id(self):

        return (
            "KIT-"
            +
            uuid.uuid4()
            .hex[:12]
            .upper()
        )


    def start(
        self,
        source,
        source_type,
        camera_id,
    ):

        session_id = (
            self._new_session_id()
        )


        STORE.create_session(
            session_id=
                session_id,

            source_type=
                source_type,

            source_path=
                str(source),

            camera_id=
                camera_id,
        )


        stop_event = (
            threading.Event()
        )


        with self.lock:

            self.stop_events[
                session_id
            ] = stop_event


        self.executor.submit(
            self._run,
            session_id,
            source,
            stop_event,
        )


        return (
            session_id
        )


    def _run(
        self,
        session_id,
        source,
        stop_event,
    ):

        try:

            pipeline = KitchenPipeline(
                session_id,
                stop_event,
            )


            pipeline.run(
                source
            )


        except Exception as exc:

            STORE.close_all_violations(
                session_id
            )


            STORE.update_session(
                session_id,
                status="failed",
                error=str(exc),
            )


        finally:

            with self.lock:

                self.stop_events.pop(
                    session_id,
                    None,
                )


    def stop(
        self,
        session_id,
    ):

        with self.lock:

            event = (
                self.stop_events
                .get(
                    session_id
                )
            )


        if event is None:

            return False


        event.set()

        return True


SERVICE = KitchenService()