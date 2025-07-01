from pydantic import BaseModel


class PredictRequest(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CurrencyCode: str
    CountryCode: float
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: float
    FraudResult: int
    # Unnamed: 16 and Unnamed: 17 are likely artifacts, so omitted


class PredictResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
