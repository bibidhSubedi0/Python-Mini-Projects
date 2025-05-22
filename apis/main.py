from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pydantic import BaseModel
from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON
from fastapi.middleware.cors import CORSMiddleware



class other_details(SQLModel):
    fingers:int = Field(default = None)
    is_a_bitch:bool = Field(default = None) 


class Student(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
    extra:other_details = Field(sa_column=Column(JSON))


sqlite_file_name = "complete_app.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/students/")
def create_students(students: Student, session: Session = Depends(get_session)) -> Student:
    session.add(students)
    session.commit()
    session.refresh(students)
    return students

@app.get("/students/")
def read_students(
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=100, le=100),
) -> list[Student]:
    students = session.exec(select(Student).offset(offset).limit(limit)).all()
    return students


@app.get("/students/{students_id}")
def read_student(students_id: int, session: Session = Depends(get_session)) -> Student:
    student = session.get(Student, students_id)  # Make sure 'Student' is a model class
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@app.delete("/students/{students_id}")
def delete_students(students_id: int, session: Session = Depends(get_session)):
    students = session.get(Student, students_id)
    if not students:
        raise HTTPException(status_code=404, detail="Student not found")
    session.delete(students)
    session.commit()
    return {"ok": True}

@app.put("/students/{student_id}")
def update_student(student_id: int, updated: Student, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    student.name = updated.name
    student.age = updated.age
    student.extra = updated.extra

    session.add(student)
    session.commit()
    session.refresh(student)
    return student

@app.get("/students/search/")
def search_students(
    name: Optional[str] = None,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    session: Session = Depends(get_session)
):
    query = select(Student)
    if name:
        query = query.where(Student.name.contains(name))
    if min_age is not None:
        query = query.where(Student.age >= min_age)
    if max_age is not None:
        query = query.where(Student.age <= max_age)

    results = session.exec(query).all()
    return results
