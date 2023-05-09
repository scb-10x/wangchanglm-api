docker build -t steerapi/xglm .
docker run -p 7860:7860 -v `pwd`/cache:/home/user/.cache -t steerapi/xglm