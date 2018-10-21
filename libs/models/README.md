# Models and Modules

A model combines am encoder module with a decoder module

The ModelManager will reference models which in turn references modules

Each model is implemented as a child class of the super: `BaseModel`, which enforces the interface