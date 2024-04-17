from langchain_core.pydantic_v1 import BaseModel, Field, validator


class ReviewClass(BaseModel):
    """Prediction of the user's comment, review or text in general"""

    # predicted class of the review/or comment
    pred: int # 0.0 1.0 -1.0 
