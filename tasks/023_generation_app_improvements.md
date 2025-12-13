# Generation App Improvements

**Status: ✅ COMPLETED**

## 1. Ensure queue robustness ✅

We want to make sure that the current queue system for generations is robust.
Rather than keeping the queue completely in memory, let's create a generations
db table that keeps track of an ordered queue of the generations to be
processed.

Let's also make sure that the generation queue is maintained completely server
side, that is - there's no local storage or client side memory of checking the
queue.

We can also add generation plans to the queue from other sources/scripts.

**Implementation:**

- Created `queue_db.py` with a SQLite-backed queue (`generation_queue` table)
- Queue supports pending, processing, complete, and error states
- Includes automatic cleanup of stale processing items on restart
- Frontend now polls server for queue status instead of using localStorage

## 2. Multiple models ✅

When we boot the app up, we shoul read from an `app_config.json` what model or
models to use. Each model will have a `name`, `model_id` and an `api_key` param,
and that model will be used for generation - all queue entries should also
contain a `model_id` so that we can use that model for generation.

The frontend needs to have a dropdown select added to the controls that lets the
user pick which model is to be used for generation. The model `name` \ `ids` in
this list should be bootstrapped into the page when served, and the selected
model id should be sent to the generate request to be added to the generation
queue.

**Implementation:**

- Created `model_config.py` with `ModelConfig` and `AppConfig` dataclasses
- Created `app_config.json` with 3 pre-configured models (Omni Water v1/v2, Omni
  Original)
- Model selector dropdown added to the toolbar
- Queue entries now include `model_id` field
- `generate_omni.py` updated to accept model configuration

## 3. Persisted web renderer ✅

Since every single generation request needs to create a new node/web server to
render the web view of tile, we should have one single web renderer process
that's started when the app server loads - the app server should make requests
to this web_renderer subprocess through a simple interface, and this interface
must account for the fact that multiple requests may come in at onece, so it
needs a queue manager for handling those requests to render specific tiles in
the order they're received.

**Implementation:**

- Created `web_renderer.py` with `WebRenderer` class
- Singleton pattern for global renderer instance
- Internal queue for render requests with callback-based results
- Auto-restarts web server if it crashes
- Started automatically when app server boots
