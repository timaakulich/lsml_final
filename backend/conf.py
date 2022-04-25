from pydantic import BaseSettings


class AppSettings(BaseSettings):
    ml_flow_server_url: str = None
    redis_host: str = None
    redis_port: int = 6379

    @property
    def redis_dsn(self):
        return f'redis://{self.redis_host}:{self.redis_port}/0'

    class Config:
        env_file = '.env'


settings = AppSettings()
