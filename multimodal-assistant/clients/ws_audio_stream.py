import asyncio, websockets, sys, wave

URI = "ws://localhost:8000/v1/realtime/audio"

async def main(path):
    async with websockets.connect(URI, max_size=None) as ws:
        # Send a small WAV file in chunks of ~100ms
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            nchan = wf.getnchannels()
            sw = wf.getsampwidth()
            chunk_ms = 100
            chunk_bytes = int(sr * (chunk_ms/1000.0)) * nchan * sw
            while True:
                data = wf.readframes(int(sr * (chunk_ms/1000.0)))
                if not data:
                    break
                await ws.send(data)
                try:
                    # receive any transcripts produced
                    recv = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    print(recv)
                except asyncio.TimeoutError:
                    pass
        # ask for final flush
        await ws.send("flush")
        try:
            while True:
                recv = await asyncio.wait_for(ws.recv(), timeout=0.5)
                print(recv)
        except asyncio.TimeoutError:
            pass

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
