# Next.js frontend. docker build -t resolvr-web web/
FROM node:22-slim AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ENV BER_API=http://api:7400
RUN npm run build

FROM node:22-slim
WORKDIR /web
COPY --from=build /web ./
ENV BER_API=http://api:7400
EXPOSE 7410
CMD ["npx", "next", "start", "-p", "7410", "-H", "0.0.0.0"]
