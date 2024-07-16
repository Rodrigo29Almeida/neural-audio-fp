from typing import Union
from fastapi import FastAPI
from fastAPI.app.models.models import Todo

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


todos = []

#Get all todos

@app.get("/todos")
async def get_todos():
    return {"todos": todos}


#Create a todo
@app.post("/todos")
async def create_todos(todo: Todo):
    todos.append(todo)
    return {"message": "Todo has been added"}