import urllib.request 
req=urllib.request.Request('http://localhost:8000/api/stream/test-camera-001/live') 
with urllib.request.urlopen(req, timeout=5) as r: 
    data=r.read(1024) 
    print('status', r.status)  
    print('headers', dict(r.getheaders())) 
    print('chunk len', len(data)) 
    print('chunk start', data[:80]) 
