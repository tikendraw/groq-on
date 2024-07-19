# Headers
## General
Request URL: https://api.groq.com/openai/v1/audio/transcriptions
Request Method: POST
Status Code: 200 OK
Remote Address: 104.18.2.205:443
Referrer Policy: strict-origin-when-cross-origin

## Response Headers

Access-Control-Allow-Origin:*
Alt-Svc: h3=":443"; ma=86400
Cache-Control: private, max-age=0, no-store, no-cache, must-revalidate
Cf-Cache-Status: DYNAMIC
Cf-Ray: 8a58e17a2f295fd0-MRS
Content-Encoding: br
Content-Type: application/json
Date: Fri, 19 Jul 2024 07:08:32 GMT
Server: cloudflare
Set-Cookie: __cf_bm=7bjWmmglXjQqagFWKlFi.Z66kFjMSdHKPTq.MDXsHhY-1721372912-1.0.1.1-cufOQCZDSaLiAVTBrkjIv_Rc4nv6PyfqSm6HnC0FIXzXjkbDlg3Z8oN4Pdvj3gpmEc0TaoX.KAWaAeG45DCecQ; path=/; expires=Fri, 19-Jul-24 07:38:32 GMT; domain=.groq.com; HttpOnly; Secure; SameSite=None
Vary: Origin
Via: 1.1 google
X-Ratelimit-Limit-Audio-Seconds: 7200
X-Ratelimit-Limit-Requests: 2000
X-Ratelimit-Remaining-Audio-Seconds: 7198
X-Ratelimit-Remaining-Requests: 1999
X-Ratelimit-Reset-Audio-Seconds: 1s
X-Ratelimit-Reset-Requests: 43.2s
X-Request-Id: req_01j34x4ap6f34vyq7t7vzx3f9f

## Request Headers

:authority: api.groq.com
:method: POST
:path: /openai/v1/audio/transcriptions
:scheme: https
Accept: application/json
Accept-Encoding: gzip, deflate, br, zstd
Accept-Language: en-US,en;q=0.9
Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Imp3ay1saXZlLTEyYTkzYTc0LWE3MWItNDJlMi1hODJiLTVjZTIyNjkxYTZkZSIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsicHJvamVjdC1saXZlLWRkYzcxNDRlLTE2MTYtNDYzMy1iMDU4LTUxNjM5M2VlMGUxNSJdLCJleHAiOjE3MjEzNzMxMzYsImh0dHBzOi8vc3R5dGNoLmNvbS9zZXNzaW9uIjp7ImlkIjoic2Vzc2lvbi1saXZlLWUyOTEyNGE1LTNkN2YtNDZjNC05ZDhiLWFlODFmY2VjODAxZiIsInN0YXJ0ZWRfYXQiOiIyMDI0LTA3LTE2VDA2OjM4OjM4WiIsImxhc3RfYWNjZXNzZWRfYXQiOiIyMDI0LTA3LTE5VDA3OjA3OjE2WiIsImV4cGlyZXNfYXQiOiIyMDI0LTA4LTE1VDA2OjM4OjM4WiIsImF0dHJpYnV0ZXMiOnsidXNlcl9hZ2VudCI6IiIsImlwX2FkZHJlc3MiOiIifSwiYXV0aGVudGljYXRpb25fZmFjdG9ycyI6W3sidHlwZSI6Im9hdXRoIiwiZGVsaXZlcnlfbWV0aG9kIjoib2F1dGhfZ29vZ2xlIiwibGFzdF9hdXRoZW50aWNhdGVkX2F0IjoiMjAyNC0wNy0xNlQwNjozODozOFoiLCJnb29nbGVfb2F1dGhfZmFjdG9yIjp7ImlkIjoib2F1dGgtdXNlci1saXZlLTQ2NTAxODA3LTcwZWUtNDRlNC1iZWEyLWJhMTQ3OTgzNTVhYyIsInByb3ZpZGVyX3N1YmplY3QiOiIxMDM5MDI1ODM0MDY1MzY4ODE0OTAifX1dfSwiaWF0IjoxNzIxMzcyODM2LCJpc3MiOiJzdHl0Y2guY29tL3Byb2plY3QtbGl2ZS1kZGM3MTQ0ZS0xNjE2LTQ2MzMtYjA1OC01MTYzOTNlZTBlMTUiLCJuYmYiOjE3MjEzNzI4MzYsInN1YiI6InVzZXItbGl2ZS02ZWNjMmQ5Mi1hNjE5LTQ5NzgtYjYwZS00MjcyOWVhZTZkNDYifQ.saKAyZ-rPBOSsQ-Ln0znHK8NjzIHW16zwvySVvRBhlPvsodSrY7FE4fyjrVA-wwaW_jdIZfpNtrkULu6qCyhR0lWgqfaIIKuTsV3zu2c362QrQfHQOWVP7OSNqVCkJXEVqX54dG01Ful2gVL8J0gX6IqhISUfQV2C6qEBy-r34d5jEQaQjgANCtKC0TThE62TL3VQ-GbqNgYI5Kd-BFLktWVdjbg9_mfvS6k60-rZ7ClPqubTamh9SROdtKbP_BKeTX9_p-zoMo0wLc6YFq6jMVm5gS-BUnTO9b7FRFrIALaKo4GxMDfX5JNzo2Ffys6XuD9PZkyvli9GlotDnqpGg
Content-Length: 37395
Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryLU5f6c0zh0y28kXz
Dnt: 1
Groq-App: chat
Groq-Organization: org_01hs882xr6e6dvkxm1q5pj7gz3
Origin: https://groq.com
Priority: u=1, i
Referer: https://groq.com/
Sec-Ch-Ua: "Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"
Sec-Ch-Ua-Mobile: ?0
Sec-Ch-Ua-Platform: "Linux"
Sec-Fetch-Dest: empty
Sec-Fetch-Mode: cors
Sec-Fetch-Site: same-site
User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36
X-Stainless-Arch: unknown
X-Stainless-Lang: js
X-Stainless-Os: Unknown
X-Stainless-Package-Version: 0.4.0
X-Stainless-Runtime: browser:chrome
X-Stainless-Runtime-Version: 126.0.0

# Form Data

## form data parsed
model: whisper-large-v3
file: (binary)

## form data source

------WebKitFormBoundaryLU5f6c0zh0y28kXz
Content-Disposition: form-data; name="model"

whisper-large-v3
------WebKitFormBoundaryLU5f6c0zh0y28kXz
Content-Disposition: form-data; name="file"; filename="groq-home-audio.webm"
Content-Type: application/octet-stream

<some binary code here>
------WebKitFormBoundaryLU5f6c0zh0y28kXz--


# Response

{
    "text": " Hello, hello, hello.",
    "x_groq": {
        "id": "req_01j34x4ap6f34vyq7t7vzx3f9f"
    }
}

# Preview

{
  "text": " Hello, hello, hello.",
  "x_groq": {
    "id": "req_01j34x4ap6f34vyq7t7vzx3f9f"
  }
}