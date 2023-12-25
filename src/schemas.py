from pydantic import BaseModel, Field


class ModelInput(BaseModel):
    """A class to represent the input of a model."""

    name: str


class ModelConfig(BaseModel):
    """A class to represent the configuration of a model."""

    name: str = Field(..., alias="model_name")
    inputs: list[ModelInput]
    teacher: ModelInput
    date_tag: ModelInput

    @property
    def inputs_tags(self) -> list[str]:
        """Get the names of the inputs."""
        return [input_.name for input_ in self.inputs]

    @property
    def teacher_tag(self) -> str:
        """Get the name of the teacher."""
        return self.teacher.name
