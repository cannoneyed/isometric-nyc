## Isometric city whitebox generation

We will implement a system to generate isometric city whitebox images from the data in the map_data directory.

* Maybe we should use this repo? https://github.com/georocket/new-york-city-model-enhanced

### ASG TUM

* This site looks very promising: https://www.asg.ed.tum.de/gis/projekte/new-york-city-3d/
  - 3D City DB: https://github.com/3dcitydb
  - Download zip of whole city here: http://www.3dcitydb.net/3dcitydb/fileadmin/public/datasets/NYC/NYC_buildings_CityGML_LoD2/NYC_Buildings_LoD2_CityGML.zip

Installing Docker
Installing the 3DCityDB docker image: https://docs.3dcitydb.org/1.1/3dcitydb/docker

```bash
docker pull 3dcitydb/3dcitydb-pg
```

```bash
docker run --name 3dciytdb -p 5432:5432 -d \
    -e POSTGRES_PASSWORD=isometric-nyc \
    -e SRID=25832 \
    -e POSTGRES_DB=isometric-nyc \
    -e POSTGRES_USER=isometric-nyc \
    -e POSTGIS_SFCGAL=true \
    3dcitydb/3dcitydb-pg
```

Docker Container ID: 61c785d1d8c93cb6e15d9c317cc2560a53ed8a93325db7f295fc1f22464719d3


Now installing the `citdyb-tool` - https://github.com/3dcitydb/citydb-tool

Also installing as a docker image!

```
docker pull 3dcitydb/citydb-tool
```

From the github repo:

> The Docker image exposes the commands of the citydb-tool, as described in the usage section. The environment variables listed below can be used to specify a 3DCityDB v5 connection. To exchange data with the container, mount a host folder to /data inside the container.

```
docker run --rm --name citydb-tool -it \
    -e CITYDB_HOST=host.docker.internal \
    -e CITYDB_PORT=5432 \
    -e CITYDB_NAME=isometric-nyc \
    -e CITYDB_USERNAME=isometric-nyc \
    -e CITYDB_PASSWORD=isometric-nyc \
    -v /Users/andycoenen/cannoneyed/isometric-nyc/map_data/3dcitydb:/data \
    3dcitydb/citydb-tool connect
```

Connection successful! Let's try importing the data

```
docker run --rm --name citydb-tool -it \
    -e CITYDB_HOST=host.docker.internal \
    -e CITYDB_PORT=5432 \
    -e CITYDB_NAME=isometric-nyc \
    -e CITYDB_USERNAME=isometric-nyc \
    -e CITYDB_PASSWORD=isometric-nyc \
    -v /Users/andycoenen/cannoneyed/isometric-nyc/map_data/3dcitydb:/data \
    3dcitydb/citydb-tool import citygml /data/NYC_Buildings_LoD2_CityGML.gml
```
