# Kestrel Home — Returns Risk: decision memo

## Decision
Use the model as a **risk-ranking layer before dispatch**, not as an automatic “return/no-return” oracle.
For next week, route the higher-risk orders to a confirmation-call workflow and reserve hard holds for a separately approved high-risk band.

## Number
The historical return rate after removing exact partner-feed re-import duplicates is 11.4%.
On the July–September dispatch snapshot, the model's average predicted return rate is 10.5%.

## Rupees
At about 700 orders/month, the model-implied gross return-cost exposure is approximately
**₹84,488/month** (700 × 0.105 × ₹1,150).

The policy says a pre-dispatch confirmation call costs ₹45 and the spring pilot prevented about 35% of returns on called orders.
Using a conservative operating threshold of 0.112 (the call-cost break-even probability is about 11.2%):
- about 195 calls/month would be triggered;
- estimated returns among those called orders: 47.7;
- estimated returns prevented: 16.7;
- gross avoided return cost: ₹19,205;
- call cost: ₹8,775;
- estimated net benefit: **₹10,430/month**.

This is a scenario estimate, not a measured production saving. The data pack does not give the monetary value of a cancelled held order, so I did not invent a hold-cost number.

## What to do next week
1. Run the score before dispatch.
2. Use a confirmation call for the risk band above ~0.11.
3. Keep a stricter high-risk band for manual hold review rather than automatically holding every flagged order.
4. Measure return rate, cancellation-after-hold rate, call completion, and prevented-return rate weekly.
5. Recalibrate the threshold after two weeks of observed outcomes.

