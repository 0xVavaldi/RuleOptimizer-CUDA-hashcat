nvcc --shared -o librules.so rules.cu -Xcompiler "-fPIC" --cudart static -arch=sm_80 --extended-lambda
go mod tidy
go build -ldflags="-r . -s -w"
