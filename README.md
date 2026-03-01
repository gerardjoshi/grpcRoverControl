# grpcRoverControl
Base Code for controlling a Raspberry PI controlled Rover with L298N motor drivers controlling regular/mechanum wheels for omnidirectional control using gRPC protocol.

Run the proto init line to ensure that the appropriate .py files are generated to ensure the server and agent_client / client files work 

```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. your_file.proto
```













If you're reading this from 3D NITT, good luck brev.
