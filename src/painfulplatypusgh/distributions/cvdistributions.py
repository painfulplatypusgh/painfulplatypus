import secrets
import math

def uniform(a: float = 0.0, b: float = 1.0)-> float:
    """Cryptographically secure uniform sample."""
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53) # in [0, 1)
    return a + (b- a) * u

def exponentialdist(lambd: float = 1.0) -> float:
    if lambd <= 0:
        raise ValueError("Lambda must be positive")
    y = uniform(0.0, 1.0)
    x = -1.0 / lambd * math.log(y)
    return x

def poissondist(lambd: float = 1.0) -> int:
  if lambd <= 0:
    raise ValueError("Lambda must be greater than 0")
  u = uniform(0.0, 1.0)
  k = 0
  cdf = (((lambd)**k)*math.exp(-lambd))/math.factorial(k)
 
  while u > cdf:
    k += 1
    cdf += (((lambd)**k)*math.exp(-lambd))/math.factorial(k)

  return k
