"""Brisbane production-hour metrics, 05:00–17:05 daily (UTC+10)."""
from datetime import datetime, timedelta, timezone
BRISBANE = timezone(timedelta(hours=10), "Brisbane UTC+10")
START_HOUR = 5
END_HOUR = 17
END_MINUTE = 5

def brisbane_now():
    return datetime.now(BRISBANE)

def hourly_snapshot(connection, source=None, now=None):
    now = (now or brisbane_now()).astimezone(BRISBANE)
    start = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    cutoff = now.replace(hour=END_HOUR, minute=END_MINUTE, second=0, microsecond=0)
    query = 'SELECT ts, metric_utc FROM counts WHERE (metric_utc>=? OR (metric_utc IS NULL AND ts>=?))'
    params = [start.timestamp(), start.replace(tzinfo=None).isoformat()]
    if source is not None:
        query += ' AND source=?'
        params.append(source)
    slots=[]
    # H1=05:00–06:00, ... H12=16:00–17:00, H13=17:00–17:05.
    for index in range(13):
        begin=start+timedelta(hours=index)
        end=min(begin+timedelta(hours=1),cutoff)
        slots.append(dict(number=index+1,start=begin,end=end,count=0,
                          completed=bool(now>=end)))
    for ts, epoch in connection.execute(query,params):
        try:
            event = datetime.fromtimestamp(epoch,BRISBANE) if epoch is not None else datetime.fromisoformat(ts)
            if event.tzinfo is None:
                event=event.replace(tzinfo=BRISBANE)
            event=event.astimezone(BRISBANE)
            if start<=event<cutoff and event<=now:
                idx=min(12,int((event-start).total_seconds()//3600))
                slots[idx]['count']+=1
        except (TypeError,ValueError,OverflowError):
            continue
    completed=[slot for slot in slots if slot['completed']]
    best=max(completed,key=lambda s:s['count']) if completed else None
    lowest=min(completed,key=lambda s:s['count']) if completed else None
    return slots,best,lowest
