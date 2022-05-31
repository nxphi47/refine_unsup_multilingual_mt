import torch
import numpy as np
import smtplib
import traceback

import os
from fairseq.distributed import utils as distributed_utils
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# TODO: setup email notifier modules

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD", "")
GMAIL_CREDENTIAL_AVAILABLE = GMAIL_USER != "" and GMAIL_PASSWORD != ""

GMAIL_RECEIVER = os.environ.get("GMAIL_RECEIVER", "")
GMAIL_RECEIVER_AVAILABLE = GMAIL_RECEIVER != ""

EMAIL_PREFIX = os.environ.get("EMAIL_PREFIX", "")

def build_email_text(sender, receivers, subject, body):
    receivers = receivers if isinstance(receivers, (list, tuple)) else [receivers]
    email_text = """
From: %s
To: %s
Subject: %s

%s
""" % (sender, ", ".join(receivers), subject, body)
    return email_text


def send_email(receiver, email_text, **kwargs):
    if not GMAIL_CREDENTIAL_AVAILABLE:
        logger.warning(f'CANNOT SEND EMAIL NOTIFICATION as {GMAIL_CREDENTIAL_AVAILABLE=}')
        return

    try:
        sent_from = GMAIL_USER
        smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        smtp_server.ehlo()
        smtp_server.login(GMAIL_USER, GMAIL_PASSWORD)
        smtp_server.sendmail(sent_from, receiver, email_text)
        smtp_server.close()
        logger.warning(f"Email from {GMAIL_USER} to {receiver} sent successfully!")
    except Exception as ex:
        logger.warning("Email: Something went wrong….",ex)


def maybe_send_email_notification(message):
    global_rank = distributed_utils.get_global_rank()
    # rank = distributed_utils.get_data_parallel_rank()
    # logger.warning(f'{global_rank=}, {rank=}')

    email_text = build_email_text(
        sender=GMAIL_USER,
        receivers=GMAIL_RECEIVER,
        subject=f'faiseq exception: [r={global_rank}] {EMAIL_PREFIX}',
        body=message
    )
    send_email(GMAIL_RECEIVER, email_text)


def notify_on_exception(fn):
    if (GMAIL_CREDENTIAL_AVAILABLE and GMAIL_RECEIVER_AVAILABLE) and distributed_utils.get_global_rank() == 0:
        logger.warning(f'WILL SEND EMAIL NOTIFICATION UPON EXCEPTION! from {fn.__name__}')

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            global_rank = distributed_utils.get_global_rank()
            # if distributed_utils.get_global_rank() == 0:
            if not GMAIL_RECEIVER_AVAILABLE:
                logger.warning(f'[{global_rank=}]GMAIL RECEIVE NOT AVAILABLE TO SEND EMAIL')
            else:
                logger.warning(f'[{global_rank=}] Try sending email notification from {GMAIL_USER}')
                error_message = traceback.format_exc()
                email_text = build_email_text(
                    sender=GMAIL_USER,
                    receivers=GMAIL_RECEIVER,
                    subject=f'faiseq exception: [{global_rank=}] {EMAIL_PREFIX}',
                    body=error_message
                )
                send_email(GMAIL_RECEIVER, email_text)
            # else:
            #     logger.warning(f'NOT SEND EMAIL BECAUSE {distributed_utils.get_global_rank()=}')
            raise e
    # wrapper.__name__ == fn.__name__
    return wrapper

