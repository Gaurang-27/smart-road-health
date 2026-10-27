"""Bounded HTTP threads; inference always stays on the single job worker."""

import threading
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server


class Server(ThreadingMixIn, WSGIServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(60)
        return request, address

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def serve(app, host, port):
    server = make_server(host, port, app, server_class=Server, handler_class=WSGIRequestHandler)
    try:
        print("Detection API listening on http://{}:{}".format(host, port), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
