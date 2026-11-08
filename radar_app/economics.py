"""Only benefits and affordable investment ceiling; no invented implementation cost."""
import math


def calculate(p):
    bounds = {"staff": (0,40), "weeks": (0,52), "before_hours": (0,40),
              "after_hours": (0,40), "hourly_cost": (0,100000),
              "cash_share": (0,1), "discount": (0,1), "wage_growth": (0,1)}
    for key, (lo,hi) in bounds.items():
        v=p[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(f"Invalid {key}")
    adoption=p["adoption"]
    if len(adoption)!=3 or any(isinstance(a,bool) or not isinstance(a,(int,float)) or not math.isfinite(a) or not 0<=a<=1 for a in adoption):
        raise ValueError("Three adoption shares in [0,1] required")
    hours=p["staff"]*p["weeks"]*(p["before_hours"]-p["after_hours"])
    annual=hours*p["hourly_cost"]
    capacity=[annual*a*(1+p["wage_growth"])**i for i,a in enumerate(adoption)]
    cash=[v*p["cash_share"] for v in capacity]
    return {"annual_hours":hours,"annual_capacity_rub":annual,
            "capacity_flows":capacity,"cash_flows":cash,
            "budget_simple":max(0,sum(cash)),
            "budget_discounted":max(0,sum(v/(1+p["discount"])**(i+1) for i,v in enumerate(cash))),
            "viable":sum(cash)>0}


SCENARIOS={
    "conservative":dict(staff=20,weeks=48,before_hours=1,after_hours=.75,hourly_cost=800,adoption=[.25,.5,.75],cash_share=0,discount=.3,wage_growth=0),
    "base":dict(staff=40,weeks=48,before_hours=2,after_hours=.5,hourly_cost=800,adoption=[.5,1,1],cash_share=.25,discount=.2,wage_growth=0),
    "optimistic":dict(staff=40,weeks=48,before_hours=3,after_hours=.5,hourly_cost=1000,adoption=[.75,1,1],cash_share=.5,discount=.1,wage_growth=0),
}
