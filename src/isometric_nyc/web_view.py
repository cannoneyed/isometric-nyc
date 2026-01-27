import os

import webview


# This is the HTML from the previous solution, simplified for the example
def get_html_content(script_content: str):
  script_content = os.path.join(os.path.dirname(__file__), "web", "main.js")
  with open(script_content, "r") as f:
    script_content = f.read()

  html_content = f"""
  <!DOCTYPE html>
  <html>
  <head>
      <style>body {{ margin: 0; overflow: hidden; }}</style>
      <script type="importmap">
        {{
          "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "3d-tiles-renderer": "https://unpkg.com/3d-tiles-renderer@0.3.26/src/index.js"
          }
        }}
      </script>
  </head>
  <body>
      <script type="module">
  {script_content}
      </script>
  </body>
  </html>
  """
  return html_content


def load_window():
  html_content = get_html_content()
  # Create a window that loads the HTML directly
  webview.create_window(
    "Isometric NYC Viewer", html=html_content, width=800, height=600
  )
  webview.start()


if __name__ == "__main__":
  load_window()
