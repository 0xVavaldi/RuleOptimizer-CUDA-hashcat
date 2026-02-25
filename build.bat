@echo off
echo Building CUDA kernel...
nvcc --shared -o rules.dll rules.cu --cudart static --extended-lambda
if %errorlevel% neq 0 (
    echo CUDA compilation failed.
    exit /b %errorlevel%
)

echo Running go mod tidy...
go mod tidy
if %errorlevel% neq 0 (
    echo go mod tidy failed.
    exit /b %errorlevel%
)

echo Building Go binary...
go build -ldflags="-s -w"
if %errorlevel% neq 0 (
    echo Go build failed.
    exit /b %errorlevel%
)

echo Build complete.
