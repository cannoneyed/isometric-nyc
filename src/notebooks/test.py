import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import dataclasses
    import io
    import json
    import os
    import requests
    from typing import Any

    import marimo as mo
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from PIL import Image
    return (
        Any,
        Image,
        dataclasses,
        genai,
        io,
        load_dotenv,
        mo,
        os,
        requests,
        types,
    )


@app.cell
def _(GoogleMapsClient, NYCOpenDataClient, load_dotenv, os):
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    nyc_opendata_app_token = os.getenv("NYC_OPENDATA_APP_TOKEN")

    gmaps = GoogleMapsClient(google_maps_api_key)
    nyc = NYCOpenDataClient(nyc_opendata_app_token)
    return gemini_api_key, gmaps, nyc


@app.cell
def _(mo):
    address_input = mo.ui.text(label="Enter an address:")
    address_input
    return (address_input,)


@app.cell
def _(Any, address_input, dataclasses, gmaps, nyc):
    @dataclasses.dataclass
    class BuildingData:
        address: str
        coords: tuple[float, float]

        footprint_data: Any
        sv_url: str
        sat_url: str

        def debug(self):
            print(f"Address: {self.address}")
            print(f"Coordinates: {self.coords}")
            print(f"Footprint Data: {self.footprint_data}")
            print(f"Satellite URL: {self.sat_url}")
            print(f"Street View URL: {self.sv_url}")

    def get_lat_lng(address: str):
        coords = gmaps.geocode(address_input.value)
        return coords

    def get_building_data(address: str):
        lat, lng = get_lat_lng(address)
        footprint_data = nyc.get_building_footprint(lat, lng)
        sat_url = gmaps.get_satellite_image_url(lat, lng)
        sv_url = gmaps.get_street_view_image_url(lat, lng)

        data = BuildingData(
            address=address,
            coords=(lat, lng),
            footprint_data=footprint_data,
            sat_url=sat_url,
            sv_url=sv_url
        )

        return data
    return (get_building_data,)


@app.cell
def _(address_input, get_building_data):
    data = get_building_data(address_input.value)
    return (data,)


@app.cell
def _(Image, data, io, mo, requests):
    # Download Street View image
    sv_response = requests.get(data.sv_url)
    street_view_image = Image.open(io.BytesIO(sv_response.content))

    # Download Satellite image
    sat_response = requests.get(data.sat_url)
    satellite_image = Image.open(io.BytesIO(sat_response.content))

    mo.hstack([
        mo.vstack([mo.md("**Street View**"), street_view_image]),
        mo.vstack([mo.md("**Satellite View**"), satellite_image])
    ])
    return satellite_image, street_view_image


@app.cell
def _(
    Image,
    data,
    gemini_api_key,
    genai,
    satellite_image,
    street_view_image,
    types,
):
    prompt = f"""An isometric pixel art view of the New York City building at 
    {data.address} (coords: {data.coords[0]}, {data.coords[1]}).
    In the style of SimCity 4000 (better than 3000). There must be no text or other graphics on the image. There should be no streets, cars, people - just the building on a plain white background.
    Please use the satellite overhead image (A) and the street view image (B) to guide generation. Ensure you stick to the exact shape of the specified building from the reference images.
    """
    aspect_ratio = "1:1" # "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
    resolution = "1K" # "1K", "2K", "4K"

    print(prompt)

    client = genai.Client(
        api_key=gemini_api_key
    )

    # Temporarily save the images
    satellite_image.save('/tmp/satellite.png')
    satellite_image_ref = client.files.upload(file="/tmp/satellite.png")

    street_view_image.save('/tmp/street_view.png')
    street_view_image_ref = client.files.upload(file="/tmp/street_view.png")

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            prompt,
            satellite_image_ref,
            street_view_image_ref,
        ],
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution
            ),
        )
    )

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif image:= part.as_image():
            print('Saving image...')
            image.save("/tmp/output.png")

    Image.open("/tmp/output.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
