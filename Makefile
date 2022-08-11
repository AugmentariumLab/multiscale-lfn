

protos: protos/*
	protoc -I=protos --python_out=protos_compiled protos/*
	protoc -I=protos --csharp_out=protos_compiled/csharp/ protos/*