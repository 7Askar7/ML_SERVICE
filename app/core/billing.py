from fastapi import HTTPException

def charge_credits(user, db, amount=1.0):
    if user.credits < amount:
        raise HTTPException(status_code=402, detail="Not enough credits")
    user.credits -= amount
    db.commit()