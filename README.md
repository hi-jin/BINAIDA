# BinAIDA
Binary Lifter extension for High-level LLVM IR

## Fun fact
BinAIDA is pronounced as "비나이다" in Korean. "비나이다" means "I pray". We pray for your happy binary analysis. 🙏

## Usage
> `/` dir
> - `create_dataset.py`: Create binary dataset

<br>

> `csmith_docker` dir
> - `csmith_repo.py`: Interface between Docker csmith and local python.
> > Something related to Docker csmith
> > - `Dockerfile`
> > - `handle_request.sh`  
> 

## Installation
### Clone
```shell
git clone https://github.com/hi-jin/BinAIDA.git
cd BinAIDA
```

### Build Docker container
```shell
cd csmith_docker
docker build -t csmith-server .
docker run -d --name csmith-server -p 8080:8080 csmith-server
```
