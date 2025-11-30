nvcc --shared -o librules.so rules.cu -Xcompiler "-fPIC" --cudart static -arch=sm_80 --extended-lambda
go build -ldflags="-r . -s -w"
