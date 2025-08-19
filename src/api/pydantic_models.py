from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: Optional[str] = ""
    CurrencyCode: str
    CountryCode: float
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: Optional[float] = None
    TransactionStartTime: str
    PricingStrategy: Optional[float] = None
    FraudResult: Optional[int] = 0
    # Unnamed: 16 and Unnamed: 17 omitted


class PredictResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
