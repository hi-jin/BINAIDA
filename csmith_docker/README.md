# csmith docker
Create an Ubuntu environment for running the csmith program, and provide a simple endpoint.

## Build
```shell
docker build -t csmith-server .
```

## Run
```shell
docker run -d --name csmith-server -p 8080:8080 csmith-server
```
