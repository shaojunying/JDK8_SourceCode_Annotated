package com.sun.corba.se.PortableActivationIDL;


/**
* com/sun/corba/se/PortableActivationIDL/ServerNotActive.java .
* Generated by the IDL-to-Java compiler (portable), version "3.2"
* from /Users/tester/jenkins/workspace/zulu8-build-macos-aarch64/zulu-src/corba/src/share/classes/com/sun/corba/se/PortableActivationIDL/activation.idl
* Thursday, July 15, 2021 8:28:48 AM PDT
*/

public final class ServerNotActive extends org.omg.CORBA.UserException
{
  public String serverId = null;

  public ServerNotActive ()
  {
    super(ServerNotActiveHelper.id());
  } // ctor

  public ServerNotActive (String _serverId)
  {
    super(ServerNotActiveHelper.id());
    serverId = _serverId;
  } // ctor


  public ServerNotActive (String $reason, String _serverId)
  {
    super(ServerNotActiveHelper.id() + "  " + $reason);
    serverId = _serverId;
  } // ctor

} // class ServerNotActive
