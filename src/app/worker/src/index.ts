/**
 * Minimal R2 caching worker - just adds Cache-Control headers
 */

interface Env {
	R2_BUCKET: R2Bucket;
}

const ALLOWED_ORIGINS = [
	"https://cannoneyed.com",
	"https://www.cannoneyed.com",
	"https://cannoneyed.github.io",
	"http://localhost:3000",
];

function getCorsHeaders(request: Request): Record<string, string> {
	const origin = request.headers.get("Origin") || "";
	const allowedOrigin = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
	return {
		"Access-Control-Allow-Origin": allowedOrigin,
		"Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
		"Access-Control-Allow-Headers": "*",
	};
}

const CACHE_HEADERS = {
	"Cache-Control": "public, max-age=31536000, immutable",
};

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const CORS_HEADERS = getCorsHeaders(request);

		// Handle CORS preflight
		if (request.method === "OPTIONS") {
			return new Response(null, { headers: CORS_HEADERS });
		}

		// Only allow GET/HEAD
		if (request.method !== "GET" && request.method !== "HEAD") {
			return new Response("Method not allowed", { status: 405, headers: CORS_HEADERS });
		}

		// Get the path (remove leading slash)
		const url = new URL(request.url);
		const key = url.pathname.slice(1);

		if (!key) {
			return new Response("Not found", { status: 404, headers: CORS_HEADERS });
		}

		// Check Cloudflare edge cache first
		// Use just the URL as cache key (no Origin header variance)
		const cache = caches.default;
		const cacheKey = new Request(url.toString(), { method: "GET" });

		let response = await cache.match(cacheKey);
		if (response) {
			// Add CORS headers fresh (not from cache)
			const headers = new Headers(response.headers);
			headers.set("Access-Control-Allow-Origin", CORS_HEADERS["Access-Control-Allow-Origin"]);
			headers.set("Access-Control-Allow-Methods", CORS_HEADERS["Access-Control-Allow-Methods"]);
			headers.set("Access-Control-Allow-Headers", CORS_HEADERS["Access-Control-Allow-Headers"]);
			return new Response(response.body, { headers });
		}

		// Cache miss - fetch from R2
		const object = await env.R2_BUCKET.get(key);

		if (!object) {
			return new Response("Not found", { status: 404, headers: CORS_HEADERS });
		}

		// Create response WITHOUT CORS headers for caching
		const responseToCache = new Response(object.body, {
			headers: {
				"Content-Type": object.httpMetadata?.contentType || "application/octet-stream",
				"ETag": object.etag,
				...CACHE_HEADERS,
			},
		});

		// Store in edge cache (don't await - fire and forget)
		cache.put(cacheKey, responseToCache.clone());

		// Return with CORS headers added
		const headers = new Headers(responseToCache.headers);
		headers.set("Access-Control-Allow-Origin", CORS_HEADERS["Access-Control-Allow-Origin"]);
		headers.set("Access-Control-Allow-Methods", CORS_HEADERS["Access-Control-Allow-Methods"]);
		headers.set("Access-Control-Allow-Headers", CORS_HEADERS["Access-Control-Allow-Headers"]);
		response = new Response(responseToCache.body, { headers });

		return response;
	},
};
