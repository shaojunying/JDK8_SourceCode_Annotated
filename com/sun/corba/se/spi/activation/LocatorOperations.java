package com.sun.corba.se.spi.activation;


/**
* com/sun/corba/se/spi/activation/LocatorOperations.java .
* Generated by the IDL-to-Java compiler (portable), version "3.2"
* from /Users/tester/jenkins/workspace/zulu8-build-macos-aarch64/zulu-src/corba/src/share/classes/com/sun/corba/se/spi/activation/activation.idl
* Thursday, July 15, 2021 8:28:48 AM PDT
*/

public interface LocatorOperations 
{

  // Starts the server if it is not already running.
  com.sun.corba.se.spi.activation.LocatorPackage.ServerLocation locateServer (int serverId, String endPoint) throws com.sun.corba.se.spi.activation.NoSuchEndPoint, com.sun.corba.se.spi.activation.ServerNotRegistered, com.sun.corba.se.spi.activation.ServerHeldDown;

  // Starts the server if it is not already running.
  com.sun.corba.se.spi.activation.LocatorPackage.ServerLocationPerORB locateServerForORB (int serverId, String orbId) throws com.sun.corba.se.spi.activation.InvalidORBid, com.sun.corba.se.spi.activation.ServerNotRegistered, com.sun.corba.se.spi.activation.ServerHeldDown;

  // get the port for the endpoint of the locator
  int getEndpoint (String endPointType) throws com.sun.corba.se.spi.activation.NoSuchEndPoint;

  // to pick a particular port type.
  int getServerPortForType (com.sun.corba.se.spi.activation.LocatorPackage.ServerLocationPerORB location, String endPointType) throws com.sun.corba.se.spi.activation.NoSuchEndPoint;
} // interface LocatorOperations
