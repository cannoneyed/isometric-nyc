import dataclasses

from google import genai


class ImageRef:
  def __init__(self, *, id: str, index: int, path: str, description: str):
    self.id = id
    self.path = path
    self.index = f"Image {chr(ord('A') + index - 1)}"
    self.description = description.replace("IMAGE_INDEX", self.index)


@dataclasses.dataclass
class Images:
  client: genai.Client

  contents: list = dataclasses.field(default_factory=list)
  descriptions: list = dataclasses.field(default_factory=list)
  refs: dict = dataclasses.field(default_factory=dict)

  def add_image_contents(self, *, id: str, path: str, description: str):
    index = len(self.contents) + 1
    image_ref = ImageRef(id=id, index=index, path=path, description=description)
    self.refs[id] = image_ref

    ref = self.client.files.upload(file=image_ref.path)
    self.contents.append(ref)
    self.descriptions.append(image_ref.description)

  def get_index(self, id: str):
    return self.refs[id].index

  def get_path(self, id: str):
    return self.refs[id].path

  def get_descriptions(self):
    return "\n".join(self.descriptions)
