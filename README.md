# Free-model aggregator

One server, two APIs, ten simulated free-tier providers.

`main.ts` (Node 22, standard library only) serves the OpenAI Chat
Completions API and the Anthropic Messages API — each in JSON and in
server-sent events — over the providers the trial env gives it. For each
request it picks the strongest free model that meets the request's
requirements, speaks that provider's wire (OpenAI, Anthropic or Gemini),
and fails over to the next-best model when a provider refuses. The answer
comes back in the wire the client used, with the provider's text whole,
its usage counts as that wire reports them, and the stop reason mapped to
the client's family.

    node main.ts --list    # print the registered catalog and exit

The server reads the brief from `MAKEFUN_PROMPT_FILE`, the provider
directory from `GET $MAKEFUN_ENV_URL/instance`, and each provider's own
catalog route, then registers its port and catalog and polls `GET
$MAKEFUN_ENV_URL/work` until the workload is done.
